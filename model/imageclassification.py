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



class CLS_head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(config.hidden_size, config.class_num)
    def forward(self, cls_token):
        return self.linear2(self.activation(self.linear1(cls_token)))

class OPTForImageClassification(OPTForPretraining):
    def __init__(self,config,video_cfg, audio_cfg):
        super().__init__(config,video_cfg, audio_cfg)
        self.cls_head = CLS_head(config)
        
    def forward(self,batch,compute_loss=True):

        batch = defaultdict(lambda: None, batch)
        video_pixels = batch['video_pixels']
        labels = batch['labels']
        

        video_output_unmasked, video_position_embedding, video_mask_indicator = \
                    self.opt.forward_video_encoder(video_pixels, perform_mask=False)

        cls_token = video_output_unmasked[:,0]

        logits = self.cls_head(cls_token)

        if compute_loss:
            
            loss = F.cross_entropy(logits, labels, reduction='mean')
            return {'loss': loss}

        else:

            return {'logits': logits,
                    'labels': labels}



