"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

OPT for pretraining
"""
from collections import defaultdict
from logging import logMultiprocessing

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from model.pretrain import OPTForPretraining
import ipdb






class OPTForCaption(OPTForPretraining):
    
    def __init__(self, config, video_cfg, audio_cfg):
        super().__init__(config, video_cfg, audio_cfg)

        self.max_generation_len = getattr(config, 'max_generation_len', 30)
        self.beam_size  = config.beam_size
        self.decode_mode = getattr(config, 'decode_mode', 'greedy')
        self.mask_token = 103
        self.eos_token = getattr(config,'eos_token',10)
        self.cls_token = 101
        self.strengthen_two = True

    def forward_caption(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        txt_tokens = batch['txt_tokens']
        video_pixels = batch['video_pixels']
        attn_mask_txt =  (txt_tokens != 0).long()


        is_3m = 'audio_spectrograms' in batch
        if is_3m:
            audio_spectrograms = batch['audio_spectrograms']

        if compute_loss:

            video_output_unmasked, video_position_embedding, video_mask_indicator, video_labels = \
                    self.opt.forward_video_encoder(video_pixels, perform_mask=False)
            video_input, attn_masks_video = self.opt.get_multimodal_forward_input_video(video_output_unmasked, video_position_embedding, video_mask_indicator)

            if is_3m :
                audio_output_unmasked, audio_position_embedding = \
                    self.opt.forward_audio_encoder(audio_spectrograms)
                
                audio_input, attn_masks_audio = self.opt.get_multimodal_forward_input_audio(audio_output_unmasked, audio_position_embedding)
            else:
                audio_input = None
                attn_masks_audio = None
                
            caption_output, txt_labels_caption = self.opt.forward_caption_encoder(txt_tokens, attn_mask_txt, \
                                            video_input, attn_masks_video, audio_input, attn_masks_audio)
            caption_output_txt = caption_output[:, :txt_tokens.shape[1], :]
            masked_output = self._compute_masked_hidden(caption_output_txt, txt_labels_caption != -1)
            prediction_scores_caption = self.cls(masked_output)       

    
            masked_lm_loss = F.cross_entropy(prediction_scores_caption,
                                        txt_labels_caption[txt_labels_caption != -1],
                                        reduction='mean') 

            if is_3m and  self.strengthen_two:
                caption_output, txt_labels_caption = self.opt.forward_caption_encoder(txt_tokens, attn_mask_txt, \
                                                video_input, attn_masks_video,None, None)
                caption_output_txt = caption_output[:, :txt_tokens.shape[1], :]
                masked_output = self._compute_masked_hidden(caption_output_txt, txt_labels_caption != -1)
                prediction_scores_caption = self.cls(masked_output)        
                masked_lm_loss_two = F.cross_entropy(prediction_scores_caption,
                                                txt_labels_caption[txt_labels_caption != -1],
                                                reduction='mean') 
                return {'caption_loss_3m': 0.5 * masked_lm_loss + 0.5 * masked_lm_loss_two}
            
            elif is_3m:
                return {'caption_loss_3m':masked_lm_loss}
            else:
                return {'caption_loss_2m':masked_lm_loss}
            

        else:
            
            video_input = batch['video_input']
            attn_masks_video = batch['attn_masks_video']
            audio_input = batch['audio_input']
            attn_masks_audio = batch['attn_masks_audio']
            

            caption_output, _ = self.opt.forward_caption_encoder(txt_tokens, attn_mask_txt, \
                                video_input, attn_masks_video, audio_input, attn_masks_audio, perform_mask=False)
            caption_output_txt = caption_output[:, :txt_tokens.shape[1], :]
            caption_output_txt = caption_output_txt[:, -1]
            prediction_scores_caption = self.cls(caption_output_txt)  
            return prediction_scores_caption

    def forward(self, batch, compute_loss=True):
        if batch['batch_3m']['ids'] != [] :
            output_3m =  self.forward_batch(batch['batch_3m'], compute_loss)
        else:
            output_3m = {}
        if batch['batch_2m']['ids'] != []:
            output_2m = self.forward_batch(batch['batch_2m'], compute_loss)
        else:
            output_2m = {}

        return {**output_3m, **output_2m }
    
    def forward_batch(self, batch, compute_loss=True):
        if compute_loss:
            return self.forward_caption(batch ,compute_loss=True)
        else:
            return self.generate_caption(batch)
    
    def generate_caption(self, batch):
        is_3m = 'audio_spectrograms' in batch

        video_pixels = batch['video_pixels']
        video_output_unmasked, video_position_embedding, video_mask_indicator, video_labels = \
                    self.opt.forward_video_encoder(video_pixels, perform_mask=False)
        video_input, attn_masks_video = self.opt.get_multimodal_forward_input_video(video_output_unmasked, video_position_embedding, video_mask_indicator)

        audio_input=None
        attn_masks_audio = None

        batch['video_input'] = video_input
        batch['attn_masks_video'] = attn_masks_video
        batch['audio_input'] = audio_input
        batch['attn_masks_audio'] = attn_masks_audio  
    
        if self.beam_size >1:
            generated_sequences = self.decode_beam(batch)
        
        else:
            generated_sequences = self.decode_greedy(batch)
        
        if not is_3m:
            return {'generated_sequence_2m': generated_sequences}

        else:
            audio_spectrograms = batch['audio_spectrograms']
            audio_output_unmasked, audio_position_embedding = \
                    self.opt.forward_audio_encoder(audio_spectrograms)
                
            audio_input, attn_masks_audio = self.opt.get_multimodal_forward_input_audio(audio_output_unmasked, audio_position_embedding)

            ### reset beam_size to 1 (first step)
            batch['video_pixels'] = video_pixels
            batch['video_input'] = video_input
            batch['attn_masks_video'] = attn_masks_video
            batch['audio_input'] = audio_input
            batch['attn_masks_audio'] = attn_masks_audio  

            if self.beam_size >1:

                generated_sequences_3m = self.decode_beam(batch)
            
            else:
                generated_sequences_3m = self.decode_greedy(batch)

        
            return {'generated_sequence_3m_woaudio': generated_sequences,
                    'generated_sequence_3m': generated_sequences_3m }


    def decode_greedy(self, batch):

        batch_size = batch['video_pixels'].size(0)
        sents = torch.zeros((batch_size, self.max_generation_len), dtype=torch.long).fill_(self.eos_token).cuda()
        logprobs = torch.zeros(batch_size, self.max_generation_len).cuda()
        unfinished = torch.ones(batch_size, dtype=torch.bool).cuda()

        state = None
        for t in range(self.max_generation_len):
            logprobs_t = self.get_logprobs(batch, state)
        
            if self.decode_mode == 'greedy': 
                logP_t, wt = torch.max(logprobs_t, 1)
            elif self.decode_mode =='sample':
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            else:
                raise NotImplementedError
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt != self.eos_token)
            wt = wt * unfinished.type_as(wt) + (1 - unfinished.type_as(wt)) * self.eos_token
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)
            state = wt.unsqueeze(1) if state is None else torch.cat((state,wt.unsqueeze(1)),dim=1)

            if unfinished.sum() == 0:
                break
        
        return sents
    def get_logprobs(self, batch, state):

        video_pixels = batch['video_pixels']
        batch_size = video_pixels.size(0)
        masked_tokens = torch.zeros(batch_size,1, dtype = torch.long, device = video_pixels.device).fill_(self.mask_token)
        cls_token = torch.zeros(batch_size,1, dtype = torch.long, device = video_pixels.device).fill_(self.cls_token)
        txt_tokens = torch.cat((state,masked_tokens), dim=1 ) if state is not None else masked_tokens
        txt_tokens = torch.cat((cls_token,txt_tokens), dim=1 )
        
        #attn_masks_txt = (txt_tokens != self.eos_token).long()

        
        txt_len = txt_tokens.shape[1]
        batch['txt_tokens'] = txt_tokens
        logits = self.forward_caption(batch, compute_loss = False)
        return F.log_softmax(logits, dim =1 )

    def decode_beam(self, batch):
        
        beam_size = self.beam_size
        batch_size = batch['video_pixels'].size(0)

        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        state = None
        #wt = torch.zeros(batch_size, dtype=torch.long).fill_(self.BOS_token).cuda()
        outputs = []
        for t in range(self.max_generation_len):
            cur_beam_size = 1 if t == 0 else beam_size
            word_logprob = self.get_logprobs(batch, state)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != self.eos_token).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                #old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            # suppress UNK tokens in the decoding
            #candidate_logprob[:, :, candidate_logprob.size(-1) - 1] = -99999

            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            # for s in range(len(state)):
            #     state[s] = self._expand_state(batch_size, beam_size, cur_beam_size, state[s], selected_beam)




            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            

            if state is not None:
                state = self._expand_state(batch_size, beam_size, state, selected_beam)
                state = torch.cat((state,selected_words),dim = 1)
            else:
                state = selected_words

            if t == 0:
                batch['video_pixels'] = self.expand_tensor(batch['video_pixels'], beam_size)
                batch['video_input'] = self.expand_tensor(batch['video_input'], beam_size)
                batch['attn_masks_video'] = self.expand_tensor(batch['attn_masks_video'], beam_size)
                if batch['audio_input'] is not None:
                    batch['audio_input'] = self.expand_tensor(batch['audio_input'], beam_size)
                    batch['attn_masks_audio'] = self.expand_tensor(batch['attn_masks_audio'], beam_size)
                
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, self.max_generation_len))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, self.max_generation_len))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        return outputs

         

    def select(self, batch_size, beam_size, t, candidate_logprob):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob


    def _expand_state(self, batch_size, beam_size, state, selected_beam):
        seq_len = state.size(-1)
        beam = selected_beam               #beam:  Bxbeam_size     state:(B*cur_beamm_size)xLXL
        beam = beam.unsqueeze(-1)       
        
        state = torch.gather(
            state.view(batch_size, beam_size, seq_len), 1,
            beam.expand(batch_size, beam_size,seq_len)
        )
        state = state.view(-1, seq_len)
        return state


    def expand_tensor(self, tensor, size, dim=1):
        if size == 1 or tensor is None:
            return tensor
        tensor = tensor.unsqueeze(dim)
        tensor = tensor.expand(list(tensor.shape[:dim]) + [size] + list(tensor.shape[dim+1:])).contiguous()
        tensor = tensor.view(list(tensor.shape[:dim-1]) + [-1] + list(tensor.shape[dim+1:]))
        return tensor







    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

