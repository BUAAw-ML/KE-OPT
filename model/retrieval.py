"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

OPT for ITM model
"""
from collections import defaultdict


import torch
from torch import nn
from .model import OPTPreTrainedModel, OPTModel
from .pretrain import  OPTForPretraining
import torch.nn.functional as F
import ipdb
import math
import horovod.torch as hvd

class OPTForVideoTextRetrievalFast(OPTForPretraining):

    def __init__(self, config, video_cfg, audio_cfg):
        super().__init__(config,video_cfg, audio_cfg)

        self.with_vata_loss = getattr(config,'with_vata_loss',False)

    def forward(self, batch, compute_loss=True):


        sample_num_2m = len(batch['batch_2m']['ids'])
        sample_num_3m = len(batch['batch_3m']['ids'])

        if sample_num_3m > 0:
            txt_tokens = batch['batch_3m']['txt_tokens']
            video_pixels = batch['batch_3m']['video_pixels']
            audio_spectrograms = batch['batch_3m']['audio_spectrograms']

            txt_output_unmasked, txt_position_embedding, attn_mask_txt, txt_labels = \
                    self.opt.forward_txt_encoder(txt_tokens, perform_mask=False)

            video_output_unmasked, video_position_embedding, video_mask_indicator, video_labels = \
                    self.opt.forward_video_encoder(video_pixels, perform_mask=False)

            audio_output_unmasked, audio_position_embedding = \
                    self.opt.forward_audio_encoder(audio_spectrograms)
        
            if sample_num_2m > 0:
                txt_tokens = batch['batch_2m']['txt_tokens']
                video_pixels = batch['batch_2m']['video_pixels']
                
                txt_output_unmasked_2m, txt_position_embedding_2m, attn_mask_txt_2m, txt_labels = \
                        self.opt.forward_txt_encoder(txt_tokens, perform_mask=False)

                video_output_unmasked_2m, video_position_embedding_2m, video_mask_indicator, video_labels = \
                        self.opt.forward_video_encoder(video_pixels, perform_mask=False)

                txt_output_unmasked = torch.cat((txt_output_unmasked, txt_output_unmasked_2m),dim = 0)
                video_output_unmasked = torch.cat((video_output_unmasked, video_output_unmasked_2m),dim = 0)
                #txt_position_embedding = torch.cat((txt_position_embedding, txt_position_embedding_2m),dim = 0) ###(1,32,768)
                #video_position_embedding = torch.cat((video_position_embedding, video_position_embedding_2m),dim = 0)
                attn_mask_txt = torch.cat((attn_mask_txt,attn_mask_txt_2m),dim=0)
        else:
            assert sample_num_2m > 0, 'empty batch'

            txt_tokens = batch['batch_2m']['txt_tokens']
            video_pixels = batch['batch_2m']['video_pixels']
                
            txt_output_unmasked, txt_position_embedding, attn_mask_txt, txt_labels = \
                        self.opt.forward_txt_encoder(txt_tokens, perform_mask=False)

            video_output_unmasked, video_position_embedding, video_mask_indicator, video_labels = \
                        self.opt.forward_video_encoder(video_pixels, perform_mask=False)

        # if self.use_multimodal_encoder:
        #     txt_output_unmasked = self.opt.get_multimodal_forward_input_txt(txt_output_unmasked, txt_position_embedding)
        #     txt_output_unmasked = self.opt.forward_retrieval_encoder(txt_input = txt_output_unmasked, 
        #                                                          attn_masks_txt = attn_mask_txt)
        #     video_output_unmasked, attn_masks_video = self.opt.get_multimodal_forward_input_video(video_output_unmasked, video_position_embedding, video_mask_indicator)
        #     video_output_unmasked = self.opt.forward_retrieval_encoder(video_input = video_output_unmasked, 
        #                                                           attn_masks_video = attn_masks_video)

        #     if sample_num_3m > 0:
        #         audio_output_unmasked, attn_masks_audio = self.opt.get_multimodal_forward_input_audio(audio_output_unmasked, audio_position_embedding)
        #         audio_output_unmasked = self.opt.forward_retrieval_encoder(audio_input = audio_output_unmasked, 
        #                                                                attn_masks_audio = attn_masks_audio)
                
        #         va_multimodal_output = self.opt.forward_retrieval_encoder(video_input = video_output_unmasked[:sample_num_3m], 
        #                                                               attn_masks_video = attn_masks_video[:sample_num_3m],
        #                                                               audio_input = audio_output_unmasked, 
        #                                                               attn_masks_audio = attn_masks_audio)

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
        feat_t = F.normalize(feat_t,dim=-1)
        feat_v = self.contra_head_v(cls_token_v)
        feat_v = F.normalize(feat_v,dim=-1)

        if sample_num_3m > 0:
            cls_token_a = audio_output_unmasked[:,0]
            feat_a = self.contra_head_a(cls_token_a)
            feat_a = F.normalize(feat_a,dim=-1)
            feat_va = self.contra_head_va_fuse(torch.cat([cls_token_v[:sample_num_3m], cls_token_a],dim=-1))
            feat_va = F.normalize(feat_va,dim=-1)

            # if self.use_multimodal_encoder:
            #     feat_va = va_multimodal_output[:,0]
            #     feat_va = self.contra_head_va(feat_va)
            #     feat_va = F.normalize(feat_va,dim=-1)
            # else:
            #     if self.va_fusion_mode == 'addition':
            #         feat_va = feat_v[:sample_num_3m] + feat_a
            #         feat_va = F.normalize(feat_va,dim=-1)
            #     elif self.va_fusion_mode == 'product':
            #         feat_va = feat_v[:sample_num_3m] * feat_a
            #         feat_va = F.normalize(feat_va,dim=-1)                
            #     elif self.va_fusion_mode == 'concate':
            #         feat_va = self.contra_head_va_fuse(torch.cat([cls_token_v[:sample_num_3m], cls_token_a],dim=-1))
            #         feat_va = F.normalize(feat_va,dim=-1)
                    
        if compute_loss:

            loss_dict = {}
            loss_dict['contra_loss_tv']  =  self.contrastive_loss(feat_t, feat_v)
            if sample_num_3m > 0:
                if self.with_vata_loss:
                    loss_dict['contra_loss_va'] = 0.1 * self.contrastive_loss(feat_v[:sample_num_3m], feat_a)
                    loss_dict['contra_loss_ta'] = 0.1 * self.contrastive_loss(feat_t[:sample_num_3m], feat_a)
                loss_dict['contra_loss_t_va'] = self.contrastive_loss(feat_t, torch.cat((feat_va,feat_v[sample_num_3m:]),dim=0))

            return loss_dict
        else:
            evaluation_dict = {}
            evaluation_dict['feat_t'] = feat_t
            evaluation_dict['feat_v'] = feat_v
            if sample_num_3m > 0:
                evaluation_dict['feat_a'] = feat_a
                evaluation_dict['feat_va'] = feat_va

            return evaluation_dict
    
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



class OPTForVideoTextRetrievalSlow(OPTForPretraining):
    
    def __init__(self, config,video_vfg):
        super().__init__(config,video_vfg)

    # def forward(self, txt,video):
    #     return self.eval_forward_slow_retrieval(txt,video)
    def forward(self, batch, compute_loss=True, pre=False, normalize=False):
        return self.forward_slow_retrieval(batch, compute_loss=compute_loss, pre=pre, \
                        normalize=normalize)
        

# class OPTForVideoTextRetrievalSlow(OPTForPretraining):
    
#     def __init__(self, config,video_vfg):
#         super().__init__(config,video_vfg)

#     def forward(self, batch, compute_loss=True):
#         txt_tokens = batch['txt_tokens']
#         video_pixels = batch['video_pixels']
#         attn_masks_txt = (txt_tokens!=0).long()
        
#         if compute_loss:
            
#             txt_embedding, attn_masks_txt, video_output, attn_masks_video,_ = self.opt.get_multimodal_forward_input(txt_tokens, video_pixels)

#             group_txt_embedding = []
#             group_video_output = []
#             group_txt_mask = []
#             group_video_mask = []
#             batch_size = txt_embedding.shape[0]

#             neg_sample_size = batch_size
#             for i in range(batch_size):
#                 for j in range(neg_sample_size):
#                     group_txt_embedding.append(txt_embedding[i].unsqueeze(0))
#                     group_video_output.append(video_output[j].unsqueeze(0))
#                     group_txt_mask.append(attn_masks_txt[i].unsqueeze(0))
#                     group_video_mask.append(attn_masks_video[j].unsqueeze(0))
#             txt_embedding = torch.cat(group_txt_embedding,dim=0)
#             video_output = torch.cat(group_video_output,dim=0)
#             attn_masks_txt = torch.cat(group_txt_mask,dim=0)
#             attn_masks_video = torch.cat(group_video_mask,dim=0)

#             multimodal_output = self.opt.multimodal_forward(txt_embedding, attn_masks_txt, video_output, attn_masks_video, multimodal_uniattn=False)
            
#             cls_token_tv = multimodal_output[:,0]
#             batch_size = txt_tokens.shape[0]
#             vtm_scores = self.match_head(cls_token_tv)
#             score_matrix = vtm_scores.reshape(batch_size,batch_size)
#             score_matrix = score_matrix / self.match_temp
#             matrix1 = -F.log_softmax(score_matrix, dim=1)
#             loss1 = matrix1.diag()
#             matrix2 = -F.log_softmax(score_matrix, dim=0)
#             loss2 = matrix2.diag()
#             total_loss = torch.mean(torch.cat((loss1,loss2), dim=0))
#             return total_loss
        
#         else:
#             sequence_output = self.opt(txt_tokens, video_pixels, attn_masks_txt) #b,n,c
#             cls_token_tv = sequence_output[:,0]
#             batch_size = txt_tokens.shape[0]
#             vtm_scores = self.match_head(cls_token_tv)
#             return vtm_scores



        