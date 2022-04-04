"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

OPT for VQA model
"""
from collections import defaultdict

from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
import torch

from .pretrain import OPTForPretraining, OPTModel
class VQA_head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(config.hidden_size, config.answer_num)
    def forward(self, cls_token):
        return self.linear2(self.activation(self.linear1(cls_token)))

class OPTForOpenEndedVQA(OPTForPretraining):
    def __init__(self,config,video_cfg, audio_cfg):
        super().__init__(config,video_cfg, audio_cfg)
        self.vqa_head = VQA_head(config)
        self.strengthen_two = True
    def forward(self,batch,compute_loss=True):
        if batch['batch_3m']['ids'] != [] :
            output_3m =  self.forward_batch(batch['batch_3m'], compute_loss)
        else:
            output_3m = {}
        if batch['batch_2m']['ids'] != []:
            output_2m = self.forward_batch(batch['batch_2m'], compute_loss)
        else:
            output_2m = {}

        return {**output_3m, **output_2m }

    def forward_batch(self,batch,compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        txt_tokens = batch['txt_tokens']  
        video_pixels = batch['video_pixels']
        answers = batch['answers']


        txt_output_unmasked, txt_position_embedding, attn_mask_txt, txt_labels = \
                    self.opt.forward_txt_encoder(txt_tokens, perform_mask=False)

        video_output_unmasked, video_position_embedding, video_mask_indicator, video_labels = \
                    self.opt.forward_video_encoder(video_pixels, perform_mask=False)

        txt_input = self.opt.get_multimodal_forward_input_txt(txt_output_unmasked, txt_position_embedding)
        video_input, attn_masks_video = self.opt.get_multimodal_forward_input_video(video_output_unmasked, video_position_embedding, video_mask_indicator)

        is_3m = 'audio_spectrograms' in batch
        if is_3m:
            audio_spectrograms = batch['audio_spectrograms']        

            audio_output_unmasked, audio_position_embedding = \
                    self.opt.forward_audio_encoder(audio_spectrograms)
                
            audio_input, attn_masks_audio = self.opt.get_multimodal_forward_input_audio(audio_output_unmasked, audio_position_embedding)
        else:
            audio_input=None
            attn_masks_audio = None

        multimodal_output = self.opt.forward_multimodal_encoder(txt_input, attn_mask_txt, \
                                                video_input, attn_masks_video, audio_input, attn_masks_audio)

        cls_token_multimodal = multimodal_output[:,0]

        logits = self.vqa_head(cls_token_multimodal)

        if compute_loss:
            
            loss = F.cross_entropy(logits, answers, reduction='mean')
            if is_3m and self.strengthen_two:
                multimodal_output = self.opt.forward_multimodal_encoder(txt_input, attn_mask_txt, \
                                                video_input, attn_masks_video, None, None)

                cls_token_multimodal = multimodal_output[:,0]

                logits = self.vqa_head(cls_token_multimodal)
                loss_2m = F.cross_entropy(logits, answers, reduction='mean')
                return {'vqa_loss_3m':0.5*loss+0.5*loss_2m}
            elif is_3m:
                return  {'vqa_loss_3m':loss}
            else:
                return  {'vqa_loss_2m':loss}
            
        else:
            if is_3m:
                multimodal_output = self.opt.forward_multimodal_encoder(txt_input, attn_mask_txt, \
                                                video_input, attn_masks_video, None, None)

                cls_token_multimodal = multimodal_output[:,0]

                logits_woaudio = self.vqa_head(cls_token_multimodal)

                return {'logits_3m': logits,
                        'logits_3m_woaudio': logits_woaudio}
            else:
                return {'logits_2m': logits}


