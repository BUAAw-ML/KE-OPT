from collections import defaultdict

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from data import retrieval

from .transformer import GELU
from .model import OPTModel, OPTPreTrainedModel
import ipdb
import numpy as np
import random
from horovod import torch as hvd
from utils.logger import LOGGER
##ss
##ss
class MLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, share_embedding = True):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = GELU()
        self.layernorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.txt_encoder_weights = config.txt_encoder_weights

        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0))
        if share_embedding:
            self.decoder.weight = bert_model_embedding_weights

    def forward(self, sequence_output):
        sequence_output = self.dense(sequence_output)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.decoder(sequence_output) 

        return prediction_scores
    
    def initialize_weights(self):
        if self.txt_encoder_weights:
            bert_weight = torch.load(self.txt_encoder_weights)
            cls_head_weight = {}
            cls_head_weight['dense.weight']  = bert_weight['cls.predictions.transform.dense.weight']
            cls_head_weight['dense.bias']  = bert_weight['cls.predictions.transform.dense.bias']
            cls_head_weight['layernorm.weight'] = bert_weight['cls.predictions.transform.LayerNorm.gamma' ]
            cls_head_weight['layernorm.bias'] =bert_weight['cls.predictions.transform.LayerNorm.beta']
            cls_head_weight['decoder.weight'] = bert_weight['cls.predictions.decoder.weight']
            cls_head_weight['decoder.bias'] = bert_weight['cls.predictions.bias']

            missing_keys, unexpected_keys = self.load_state_dict(cls_head_weight)
            LOGGER.info(f'missing_keys in cls_head : {missing_keys}')
            LOGGER.info(f'unexpected_keys in cls_head : {unexpected_keys}')
            del(bert_weight)
            del(cls_head_weight)

class MVM_head(nn.Module):
    def __init__(self, config, video_cfg):
        super().__init__()
        output_dim = video_cfg['patch_size']**2*3
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = GELU()
        self.layernorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(config.hidden_size, output_dim)
    def forward(self, sequence_output):
        sequence_output = self.linear1(sequence_output)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.layernorm(sequence_output)
        regression_results = self.linear2(sequence_output) 

        return regression_results

class Contra_head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.contra_dim, bias=False)

    def forward(self, cls_token):
        return self.linear(cls_token)

class Match_head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = GELU()
        self.layernorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(config.hidden_size, 2)
    def forward(self, cls_token):
        return self.linear2(self.layernorm(self.activation(self.linear1(cls_token))))

class OPTForPretraining(OPTPreTrainedModel):
    """ OPT pretraining """
    def __init__(self, config, video_cfg, audio_cfg):
        super().__init__(config, video_cfg, audio_cfg)

        self.opt = OPTModel(config, video_cfg, audio_cfg)
        self.cls = MLMHead(
            config, self.opt.txt_embeddings.word_embeddings.weight,share_embedding=True)
        
        self.mvm_head = MVM_head(config, video_cfg)
        self.match_head = Match_head(config)
        self.contra_head_v = Contra_head(config)
        self.contra_head_t = Contra_head(config)
        #self.contra_head_a = Contra_head(config)
        #self.contra_head_va = Contra_head(config)
        self.contra_head_va_fuse = nn.Linear(2*config.hidden_size, config.contra_dim, bias=False)
        self.contra_temp = nn.Parameter(torch.tensor(0.07))
        #self.match_temp = nn.Parameter(torch.tensor(0.07))

        self.match_mode = config.match_mode
        self.pretrain_match_mode = config.pretrain_match_mode
        self.video_cfg = video_cfg
        self.contra_sync = True
        #self.contra_filter = getattr(config,'contra_filter',False)
        self.strengthen_two = config.strengthen_two
        self.apply(self.init_weights)
        self.opt.initialize_weights()
        self.cls.initialize_weights()


    def forward(self,batch, task, compute_loss=True):
        if 'Three' in task: 
            return self.forward_3m(batch['batch_3m'], task, compute_loss)
        elif 'Two' in task: 
            return self.forward_2m(batch['batch_2m'], task, compute_loss)



    def forward_2m(self, batch, task, compute_loss=True):
        #### pretraining forward function ##
        if compute_loss:
            loss_dict = {}
        else:
            evaluation_dict = {}

        batch = defaultdict(lambda: None, batch)
        txt_tokens = batch['txt_tokens']
        video_pixels = batch['video_pixels']

        task = task.split('_')
        assert 'contraTwo' in task

        
            
        txt_output_unmasked, txt_position_embedding, attn_mask_txt, txt_labels = \
                    self.opt.forward_txt_encoder(txt_tokens, perform_mask=False)

        video_output_unmasked, video_position_embedding, video_mask_indicator, video_labels = \
                    self.opt.forward_video_encoder(video_pixels, perform_mask=False)
        
        #### forward contra 
        cls_token_t = txt_output_unmasked[:,0] 
        if self.opt.video_encoder_type == 'videoswin':
            cls_token_v = torch.mean(video_output_unmasked, dim=1)  ### first token of vide output is random initialized 
        elif self.opt.video_encoder_type.startswith('vit'):
            if self.opt.video_encoder_type == 'vit_global':
                cls_token_v = torch.mean(video_output_unmasked, dim=1) 
            elif self.opt.video_encoder_type == 'vit_local':
                b , n, hidden_size = video_output_unmasked.shape
                frame_num = self.video_cfg['sample_num']
                video_output_unmasked = video_output_unmasked.reshape(b*frame_num, -1, hidden_size)[:,0].reshape(b,frame_num,hidden_size)
                cls_token_v = torch.mean(video_output_unmasked, dim=1) 
        else:
            cls_token_v = video_output_unmasked[:,0]

        feat_t = self.contra_head_t(cls_token_t)
        normalized_txt =  F.normalize(feat_t,dim=-1)
        feat_v = self.contra_head_v(cls_token_v)
        normalized_video =  F.normalize(feat_v,dim=-1)

        if compute_loss:
            loss_dict['contra_loss'] = self.contrastive_loss(normalized_txt, normalized_video)
            
        else:
            evaluation_dict['normalized_txt'] = normalized_txt
            evaluation_dict['normalized_video'] = normalized_video

        if 'mlmTwo' in task:
            #### forward mlmTwo

            txt_output_masked, txt_position_embedding, attn_mask_txt, txt_labels = \
                    self.opt.forward_txt_encoder(txt_tokens, perform_mask=True)

            txt_input = self.opt.get_multimodal_forward_input_txt(txt_output_masked, txt_position_embedding)
            video_input, attn_masks_video = self.opt.get_multimodal_forward_input_video(video_output_unmasked, video_position_embedding, video_mask_indicator)
            multimodal_output = self.opt.forward_multimodal_encoder(txt_input, attn_mask_txt, video_input, attn_masks_video)
            multimodal_output_txt = multimodal_output[:, 1 : 1 + txt_tokens.shape[1], :]
            masked_output = self._compute_masked_hidden(multimodal_output_txt, txt_labels != -1)
            prediction_scores = self.cls(masked_output)

            if compute_loss:
                masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='mean')
                loss_dict['masked_lm_loss'] = masked_lm_loss
                
                
            else:
                evaluation_dict['prediction_scores'] = prediction_scores
                evaluation_dict['txt_labels'] = txt_labels
        
        if 'matchTwo' in task:
            
            txt_input = self.opt.get_multimodal_forward_input_txt(txt_output_unmasked, txt_position_embedding)
            video_input, attn_masks_video = self.opt.get_multimodal_forward_input_video(video_output_unmasked, video_position_embedding, video_mask_indicator)
            #### forward matchTwo

            video_inputs = []
            ground_truth = []
            batch_size = video_input.shape[0]

            if self.pretrain_match_mode == 'pair_hard':
                score_matrix = torch.matmul(normalized_txt, normalized_video.permute(1,0))
                hard_positive_idxs = score_matrix.diag().sort()[1][:batch_size//2]
                score_matrix = score_matrix + -10e6*torch.eye(batch_size).to(score_matrix)
                hard_negative_idxs = score_matrix.max(dim=1)[1]
            
            for i in range(batch_size):
                if self.pretrain_match_mode == 'pair_hard':
                    is_match = (i in hard_positive_idxs)
                else:
                    is_match = random.random() > 0.5 or batch_size == 1
                if is_match:
                    video_inputs.append(video_input[i])
                    ground_truth.append(1)
                else:
                    if self.pretrain_match_mode == 'pair_hard':
                        sample_idx = hard_negative_idxs[i]
                    else:
                        sample_idx = random.choice([j for j in range(batch_size) if j!=i ])
                    video_inputs.append(video_input[sample_idx])
                    ground_truth.append(0)
            video_input = torch.stack(video_inputs, dim=0)
            multimodal_output = self.opt.forward_multimodal_encoder(txt_input, attn_mask_txt, video_input, attn_masks_video)
            cls_token_tv = multimodal_output[:,0]
            vtm_scores = self.match_head(cls_token_tv)
            ground_truth = torch.tensor(ground_truth).long().cuda()
            if compute_loss:
                match_loss = F.cross_entropy(vtm_scores, ground_truth, reduction = 'mean')
                loss_dict['match_loss'] =  match_loss
                
            else:
                evaluation_dict['vtm_scores'] = vtm_scores
                evaluation_dict['ground_truth'] = ground_truth
        
        if 'unimlmTwo' in task:
            #### forward caption
            video_input, attn_masks_video = self.opt.get_multimodal_forward_input_video(video_output_unmasked, video_position_embedding, video_mask_indicator)
            caption_output, txt_labels_caption = self.opt.forward_caption_encoder(txt_tokens, attn_mask_txt, video_input, attn_masks_video)
            caption_output_txt = caption_output[:, :txt_tokens.shape[1], :]
            masked_output = self._compute_masked_hidden(caption_output_txt, txt_labels_caption != -1)
            prediction_scores_caption = self.cls(masked_output)

            if compute_loss:
                masked_lm_loss_caption = F.cross_entropy(prediction_scores_caption,
                                             txt_labels_caption[txt_labels_caption != -1],
                                             reduction='mean')    
                loss_dict['masked_lm_loss_caption'] =  masked_lm_loss_caption
            else:
                evaluation_dict['prediction_scores_caption'] = prediction_scores_caption
                evaluation_dict['txt_labels_caption'] = txt_labels_caption
        
        if compute_loss:
            return loss_dict 
        else:
            return evaluation_dict



    def forward_3m(self, batch, task, compute_loss=True):
        #### pretraining forward function ##
        if compute_loss:
            loss_dict = {}
        else:
            evaluation_dict = {}

        batch = defaultdict(lambda: None, batch)
        txt_tokens = batch['txt_tokens']
        video_pixels = batch['video_pixels']
        audio_spectrograms = batch['audio_spectrograms']
        task = task.split('_')
        assert 'contraThree' in task

        

        txt_output_unmasked, txt_position_embedding, attn_mask_txt, txt_labels = \
                    self.opt.forward_txt_encoder(txt_tokens, perform_mask=False)

        video_output_unmasked, video_position_embedding, video_mask_indicator, video_labels = \
                    self.opt.forward_video_encoder(video_pixels, perform_mask=False)

        audio_output_unmasked, audio_position_embedding = \
                    self.opt.forward_audio_encoder(audio_spectrograms)

        
        #### forward contra 
        cls_token_t = txt_output_unmasked[:,0] 
        if self.opt.video_encoder_type == 'videoswin':
            cls_token_v = torch.mean(video_output_unmasked, dim=1)  ### first token of vide output is random initialized 
        elif self.opt.video_encoder_type.startswith('vit'):
            if self.opt.video_encoder_type == 'vit_global':
                cls_token_v = torch.mean(video_output_unmasked, dim=1) 
            elif self.opt.video_encoder_type == 'vit_local':
                b , n, hidden_size = video_output_unmasked.shape
                frame_num = self.video_cfg['sample_num']
                video_output_unmasked = video_output_unmasked.reshape(b*frame_num, -1, hidden_size)[:,0].reshape(b,frame_num,hidden_size)
                cls_token_v = torch.mean(video_output_unmasked, dim=1) 
        else:
            cls_token_v = video_output_unmasked[:,0]

        cls_token_a = audio_output_unmasked[:,0]

        feat_t = self.contra_head_t(cls_token_t)
        feat_t = F.normalize(feat_t,dim=-1)
        feat_v = self.contra_head_v(cls_token_v)
        feat_v = F.normalize(feat_v,dim=-1)
        feat_va = self.contra_head_va_fuse(torch.cat([cls_token_v, cls_token_a],dim=-1))
        feat_va = F.normalize(feat_va,dim=-1)



        if compute_loss:
            loss_dict[f'contra_loss_tv'] = self.contrastive_loss(feat_t, feat_v)
            loss_dict[f'contra_loss_t_va'] = self.contrastive_loss(feat_t, feat_va)
            
        else:
            evaluation_dict['normalized_txt'] = feat_t
            evaluation_dict['normalized_video'] = feat_v
            evaluation_dict['normalized_va'] = feat_va

        if 'mlmThree' in task:
            #### forward mlmThree

            txt_output_masked, txt_position_embedding, attn_mask_txt, txt_labels = \
                    self.opt.forward_txt_encoder(txt_tokens, perform_mask=True)

            txt_input = self.opt.get_multimodal_forward_input_txt(txt_output_masked, txt_position_embedding)
            video_input, attn_masks_video = self.opt.get_multimodal_forward_input_video(video_output_unmasked, video_position_embedding, video_mask_indicator)
            audio_input, attn_masks_audio = self.opt.get_multimodal_forward_input_audio(audio_output_unmasked, audio_position_embedding)
            multimodal_output = self.opt.forward_multimodal_encoder(txt_input, attn_mask_txt, \
                                        video_input, attn_masks_video, audio_input, attn_masks_audio)
                                    
            multimodal_output_txt = multimodal_output[:, 1 : 1 + txt_tokens.shape[1], :]
            masked_output = self._compute_masked_hidden(multimodal_output_txt, txt_labels != -1)
            prediction_scores = self.cls(masked_output)
            ##### strengthtwo
            if self.strengthen_two:
                multimodal_output_two = self.opt.forward_multimodal_encoder(txt_input, attn_mask_txt, \
                                        video_input, attn_masks_video, None,  None)
                multimodal_output_txt_two = multimodal_output_two[:, 1 : 1 + txt_tokens.shape[1], :]
                masked_output_two = self._compute_masked_hidden(multimodal_output_txt_two, txt_labels != -1)
                prediction_scores_two = self.cls(masked_output_two)

            if compute_loss:
                masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='mean')

                loss_dict['masked_lm_loss'] = masked_lm_loss

                if self.strengthen_two:
                    masked_lm_loss_woaudio = F.cross_entropy(prediction_scores_two,
                                             txt_labels[txt_labels != -1],
                                             reduction='mean')
                    loss_dict['masked_lm_loss_woaudio'] = masked_lm_loss_woaudio
                
                
            else:
                evaluation_dict['prediction_scores'] = prediction_scores
                evaluation_dict['txt_labels'] = txt_labels
                if self.strengthen_two:
                    evaluation_dict['prediction_scores_woaudio'] = prediction_scores_two
        

        if 'unimlmThree' in task:
            #### forward caption
            video_input, attn_masks_video = self.opt.get_multimodal_forward_input_video(video_output_unmasked, \
                                                        video_position_embedding, video_mask_indicator)
            audio_input, attn_masks_audio = self.opt.get_multimodal_forward_input_audio(audio_output_unmasked, audio_position_embedding)
            caption_output, txt_labels_caption = self.opt.forward_caption_encoder(txt_tokens, attn_mask_txt, \
                                                    video_input, attn_masks_video, audio_input, attn_masks_audio)
            caption_output_txt = caption_output[:, :txt_tokens.shape[1], :]
            masked_output = self._compute_masked_hidden(caption_output_txt, txt_labels_caption != -1)
            prediction_scores_caption = self.cls(masked_output)

            if self.strengthen_two:
                caption_output_two, txt_labels_caption_two = self.opt.forward_caption_encoder(txt_tokens, attn_mask_txt, \
                                                        video_input, attn_masks_video, None, None)
                caption_output_txt_two = caption_output_two[:, :txt_tokens.shape[1], :]
                masked_output_two = self._compute_masked_hidden(caption_output_txt_two, txt_labels_caption_two != -1)
                prediction_scores_caption_two = self.cls(masked_output_two)

            if compute_loss:
                masked_lm_loss_caption = F.cross_entropy(prediction_scores_caption,
                                             txt_labels_caption[txt_labels_caption != -1],
                                             reduction='mean')    
                loss_dict['masked_lm_loss_caption'] =  masked_lm_loss_caption

                if self.strengthen_two:
                    masked_lm_loss_caption_woaudio = F.cross_entropy(prediction_scores_caption_two,
                                             txt_labels_caption_two[txt_labels_caption_two != -1],
                                             reduction='mean')    
                    loss_dict['masked_lm_loss_caption_woaudio'] =  masked_lm_loss_caption_woaudio

            else:
                evaluation_dict['prediction_scores_caption'] = prediction_scores_caption
                evaluation_dict['txt_labels_caption'] = txt_labels_caption
                if self.strengthen_two:
                    evaluation_dict['prediction_scores_caption_two'] = prediction_scores_caption_two
                    evaluation_dict['txt_labels_caption_two'] = txt_labels_caption_two
        
        if compute_loss:
            return loss_dict 
        else:
            return evaluation_dict



    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    
    def contrastive_loss(self, normalized_m1, normalized_m2):

        if self.contra_sync:
            normalized_m1 = hvd.allgather(normalized_m1)
            normalized_m2 = hvd.allgather(normalized_m2)
        score_matrix = torch.matmul(normalized_m1, normalized_m2.permute(1,0))
        score_matrix = score_matrix / self.contra_temp
        matrix1 = -F.log_softmax(score_matrix, dim=1)
        matrix2 = -F.log_softmax(score_matrix, dim=0)
        loss1 = matrix1.diag()
        loss2 = matrix2.diag()
        contra_loss = torch.mean(torch.cat((loss1,loss2), dim=0))
        return contra_loss