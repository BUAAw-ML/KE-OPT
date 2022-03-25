"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
import copy
import json
import logging
from io import open
from ntpath import join
import ipdb
import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm 
from torch.nn.modules import dropout
import math
from .transformer import OPTEncoder
from .timesformer import TimesFormerEncoder
import random
import numpy as np
from .videoswin import SwinTransformer3D
import time
from utils.logger import LOGGER
import torch.nn.functional as F

#logger = logging.getLogger(__name__)


# class OPTConfig(object):
#     """Configuration class to store the configuration of a `OPTModel`.
#     """
#     def __init__(self,
#                  vocab_size_or_config_json_file,
#                  hidden_size=768,
#                  num_hidden_layers=12,
#                  num_attention_heads=12,
#                  intermediate_size=3072,
#                  hidden_act="gelu",
#                  hidden_dropout=0.1,
#                  attention_dropout=0.1,
#                  max_position_embeddings=512,
#                  type_vocab_size=2,
#                  initializer_range=0.02):
#         """Constructs OPTConfig.
#         Args:
#             vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in
#                 `OPTModel`.
#             hidden_size: Size of the encoder layers and the pooler layer.
#             num_hidden_layers: Number of hidden layers in the Transformer
#                 encoder.
#             num_attention_heads: Number of attention heads for each attention
#                 layer in the Transformer encoder.
#             intermediate_size: The size of the "intermediate" (i.e.
#                 feed-forward) layer in the Transformer encoder.
#             hidden_act: The non-linear activation function (function or string)
#                 in the encoder and pooler. If string, "gelu", "relu" and
#                 "swish" are supported.
#             hidden_dropout: The dropout probabilitiy for all fully
#                 connected layers in the embeddings, encoder, and pooler.
#             attention_dropout: The dropout ratio for the attention
#                 probabilities.
#             max_position_embeddings: The maximum sequence length that this
#                 model might ever be used with. Typically set this to something
#                 large just in case (e.g., 512 or 1024 or 2048).
#             type_vocab_size: The vocabulary size of the `token_type_ids` passed
#                 into `OPTModel`.
#             initializer_range: The sttdev of the truncated_normal_initializer
#                 for initializing all weight matrices.
#         """
#         if isinstance(vocab_size_or_config_json_file, str):
#             with open(vocab_size_or_config_json_file,
#                       "r", encoding='utf-8') as reader:
#                 json_config = json.loads(reader.read())
#             for key, value in json_config.items():
#                 self.__dict__[key] = value
#         elif isinstance(vocab_size_or_config_json_file, int):
#             self.vocab_size = vocab_size_or_config_json_file
#             self.hidden_size = hidden_size
#             self.num_hidden_layers = num_hidden_layers
#             self.num_attention_heads = num_attention_heads
#             self.hidden_act = hidden_act
#             self.intermediate_size = intermediate_size
#             self.hidden_dropout = hidden_dropout
#             self.attention_dropout = attention_dropout
#             self.max_position_embeddings = max_position_embeddings
#             self.type_vocab_size = type_vocab_size
#             self.initializer_range = initializer_range
#         else:
#             raise ValueError("First argument must be either a vocabulary size "
#                              "(int) or the path to a pretrained model config "
#                              "file (str)")

class OPTConfig(object):
    def __init__(self,
                 config):
        
        if isinstance(config, dict):
            for key, value in config.items():
                self.__dict__[key] = value

        else:
            raise ValueError("First argument must be either a vocabulary size "
                             "(int) or the path to a pretrained model config "
                             "file (str)")
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `OPTConfig` from a
           Python dictionary of parameters."""
        config = OPTConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `OPTConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class OPTPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, OPTConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of "
                "class `OPTConfig`. To create a model from a Google "
                "pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, config, state_dict, *inputs, **kwargs):
        """
        Instantiate a OPTPreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific OPT class
        """
        # Load config
        #config = OPTConfig.from_json_file(config_file)
        # config = OPTConfig(vocab_size_or_config_json_file = config_file)
        config = OPTConfig(config)
        LOGGER.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = ({} if metadata is None
                              else metadata.get(prefix[:-1], {}))
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.')
                                              for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            LOGGER.info("Weights of {} not initialized from "
                        "pretrained model: {}".format(
                            model.__class__.__name__, str(missing_keys)))
        if len(unexpected_keys) > 0:
            LOGGER.info("Weights from pretrained model not used in "
                        "{}: {}".format(
                            model.__class__.__name__, str(unexpected_keys)))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for '
                               '{}:\n\t{}'.format(
                                   model.__class__.__name__,
                                   "\n\t".join(error_msgs)))
        return model





class OPTTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layernorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.mask_prob = getattr(config,'txt_mask_prob', 0.15)
        self.mask_token = 103
        
        self.range = [106, config.vocab_size]

    

    def forward(self, txt_tokens, perform_mask = False, txt_mask_indicator = None):
        txt_labels = None
        if perform_mask:
            txt_tokens = txt_tokens.clone() ### important, must have
            txt_tokens, txt_labels = self.perform_mask(txt_tokens, txt_mask_indicator=txt_mask_indicator)

        words_embeddings = self.word_embeddings(txt_tokens)
        position_ids = torch.arange(words_embeddings.shape[1], dtype=torch.long, device= words_embeddings.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = words_embeddings + position_embeddings 
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings, position_embeddings, txt_labels
    
    def perform_mask(self, txt_tokens, txt_mask_indicator=None):
        if txt_mask_indicator is None:
            ### generate indicator first:
            txt_mask_indicator = torch.zeros_like(txt_tokens)
            for i in range(len(txt_mask_indicator)):
                while all(txt_mask_indicator[i] == 0):
                    for j in range(1, len(txt_mask_indicator[0])):
                        if txt_tokens[i][j]!=0 and random.random() < self.mask_prob:
                            txt_mask_indicator[i][j] = 1
                
        labels = torch.zeros_like(txt_tokens).fill_(-1)
        for i in range(txt_tokens.shape[0]):
            for j in range(txt_tokens.shape[1]):
                
                if txt_mask_indicator[i][j] == 1 :
                    src_token = txt_tokens[i][j].item()
                    prob = random.random()
                    if prob < 0.8:
                        txt_tokens[i][j] = self.mask_token
                    elif prob < 0.9:
                        txt_tokens[i][j] = random.choice(list(range(*self.range)))            
                        
                    labels[i][j] = src_token

                
        return txt_tokens, labels






class TimesformerVideoEmbeddings(nn.Module):
    def __init__(self, config, video_cfg):
        super().__init__()
        self.sample_num = video_cfg['sample_num']
        self.patch_size = video_cfg['patch_size']
        self.token_length_per_frame = (video_cfg['resolution'] // self.patch_size) **2
        self.first_conv = nn.Conv2d(3, config.hidden_size, kernel_size = self.patch_size, 
                                    stride = self.patch_size, padding=0)
        self.position_embeddings = nn.Embedding(self.token_length_per_frame + 1, config.hidden_size)
        self.frame_embedding = nn.Embedding(10,config.hidden_size)  ###assert max 10 frames
        
        # self.layernorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.cls_token = nn.Parameter(0.02 * torch.randn(1, 1, config.hidden_size)) 
        self.mask_prob = getattr(config,'video_mask_prob', 0.6)
        self.mask_token = nn.Parameter(0.02 * torch.randn(config.hidden_size)) 
        self.block_masking = True        

    def forward(self, video_pixels, perform_mask = False):  ### shape Bxnx3xHxW
        b, n, c, h, w = video_pixels.shape
        video_pixels = video_pixels.reshape(b*n, c,h,w) 
        video_mask_indicator = None 
        video_pixels = self.first_conv(video_pixels)  ### B*n, 768,h,w
        video_pixels = video_pixels.permute(0,2,3,1) ### B*n, h,w,768
        video_pixels = video_pixels.reshape(b,-1,video_pixels.shape[-1])
        
        batch_size = video_pixels.shape[0]
        cls_token = self.cls_token.expand(batch_size,-1,-1)
        video_tokens = torch.cat((cls_token,video_pixels),dim=1)

        video_pos_ids = [0] + list(range(1, self.token_length_per_frame + 1)) * self.sample_num
        video_pos_ids = torch.tensor(video_pos_ids, dtype=torch.long, device=video_pixels.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(video_pos_ids)

        frame_ids = [i for i in range(self.sample_num) for j in range(self.token_length_per_frame)]
        frame_ids = torch.tensor(frame_ids, dtype=torch.long, device=video_pixels.device).unsqueeze(0)
        position_embeddings[:,1:] += self.frame_embedding(frame_ids)

        embeddings = video_tokens + position_embeddings 
        #embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)


        if perform_mask:

            video_mask_indicator = torch.zeros(video_pixels.shape[:2]).long().cuda()
            
            ### make sure every frame mask same number of tokens 
            
            if not self.block_masking:
                ### random masking 
                video_mask_indicator = video_mask_indicator.reshape(b*self.sample_num, -1)
                mask_num  = int(self.mask_prob*video_mask_indicator.shape[1])
                video_mask_indicator[:,:mask_num] = 1
                for i in range(video_mask_indicator.shape[0]):
                    shuffle_idx = torch.randperm(video_mask_indicator.shape[1])
                    video_mask_indicator[i] = video_mask_indicator[i][shuffle_idx]
            
            else:
                ###block_masking:
                h = w = int(math.sqrt(self.token_length_per_frame))
                video_mask_indicator = video_mask_indicator.reshape(b, self.sample_num, h, w)
                for i in range(b):
                    masked_h = int(self.mask_prob*h)
                    masked_w = int(self.mask_prob*w)
                    start_h = np.random.randint(0,h-masked_h)
                    start_w = np.random.randint(0,w-masked_w)
                    video_mask_indicator[i,:,start_h:start_h+masked_h, start_w:start_w+masked_w] = 1
                


            video_mask_indicator = video_mask_indicator.reshape(b,-1)

            #video_pixels[video_mask_indicator.bool()] = self.mask_token           
            #video_labels = video_pixels_raw[video_mask_indicator.bool()]
        
            cls_embedding = embeddings[:,0:1]
            res_embedding = embeddings[:,1:]
            dim = res_embedding.shape[-1]
            unmasked_idx = ~video_mask_indicator.bool()
            res_embedding = res_embedding[unmasked_idx].reshape(b,-1,dim)
            embeddings = torch.cat((cls_embedding, res_embedding),dim=1)


        return embeddings, position_embeddings, video_mask_indicator
        #return embeddings, position_embeddings, video_mask_indicator, video_labels


class VitVideoEmbeddings(nn.Module):
    def __init__(self, config, video_cfg):
        super().__init__()
        self.sample_num = video_cfg['sample_num']
        self.token_length_per_frame = (video_cfg['resolution'] // video_cfg['patch_size']) **2
        self.first_conv = nn.Conv2d(3, config.hidden_size, kernel_size = video_cfg['patch_size'], 
                                    stride = video_cfg['patch_size'], padding=0)
        self.position_embeddings = nn.Embedding(self.token_length_per_frame + 1, config.hidden_size)
        self.frame_embedding = nn.Embedding(10,config.hidden_size)  ###assert max 10 frames
        
        # self.layernorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.cls_token = nn.Parameter(0.02 * torch.randn(1, 1, config.hidden_size)) 
        self.drop_ratio = video_cfg.get('drop_ratio',0)
        self.video_encoder_type = config.video_encoder_type

    def forward(self, video_pixels):  ### shape Bxnx3xHxW
        b, n, c, h, w = video_pixels.shape
        video_pixels = video_pixels.reshape(b*n, c,h,w) 
        video_pixels = self.first_conv(video_pixels)  ### B*n, 768,h,w
        video_pixels = video_pixels.permute(0,2,3,1) ### B*n, h,w,768
        video_pixels = video_pixels.reshape(video_pixels.shape[0],-1,video_pixels.shape[-1])
        
        batch_size = video_pixels.shape[0]
        cls_token = self.cls_token.expand(batch_size,-1,-1)
        video_tokens = torch.cat((cls_token,video_pixels),dim=1)

        video_pos_ids = list(range(self.token_length_per_frame + 1)) 
        video_pos_ids = torch.tensor(video_pos_ids, dtype=torch.long, device=video_pixels.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(video_pos_ids)
        if self.video_encoder_type == 'vit_local':
            frame_ids = [i for i in range(self.sample_num) for j in range(self.token_length_per_frame + 1)]
        else:
            frame_ids = [i for i in range(self.sample_num)] 
        frame_ids = torch.tensor(frame_ids, dtype=torch.long, device=video_pixels.device).unsqueeze(0)
        frame_embeddings = self.frame_embedding(frame_ids)



        embeddings = video_tokens + position_embeddings 
        #embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        if self.training and self.drop_ratio > 0:
            embeddings_p1 = embeddings[:,0] ### cls token do not drop
            embeddings_p2 = embeddings[:,1:]
            src_len = embeddings_p2.shape[1]
            tgt_len = int((1 - self.drop_ratio) * src_len)
            tmp = list(range(src_len))
            gather_idx = [ random.sample(tmp,tgt_len) for i in range(embeddings_p2.shape[0]) ]
            for i in gather_idx:
                i.sort()
            gather_idx = torch.tensor(gather_idx).to(video_pos_ids).unsqueeze(-1).expand(-1,-1,embeddings_p2.shape[-1])
            embeddings_p2 = torch.gather(embeddings_p2, 1 , gather_idx)
            embeddings = torch.cat((embeddings_p1.unsqueeze(1), embeddings_p2),dim=1)
        return embeddings, position_embeddings, frame_embeddings



class OPTAudioEmbeddings(nn.Module):
    def __init__(self, config, audio_cfg):
        super().__init__()
       
        self.patch_size = audio_cfg['patch_size']
        self.token_length_per_frame = (audio_cfg['melbins'] // self.patch_size) * (audio_cfg['target_length'] // self.patch_size)
        self.first_conv = nn.Conv2d(1, config.hidden_size, kernel_size = self.patch_size, 
                                    stride = self.patch_size, padding=0)
        self.position_embeddings = nn.Embedding(self.token_length_per_frame + 1, config.hidden_size)
        
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.cls_token = nn.Parameter(0.02 * torch.randn(1, 1, config.hidden_size)) 
        
    def forward(self, audio_spectrograms):  ### shape Bx128x1024
        audio_spectrograms = self.first_conv(audio_spectrograms.unsqueeze(1))
        b,c,_,_=audio_spectrograms.shape
        audio_tokens = audio_spectrograms.permute(0,2,3,1).reshape(b,-1,c)        
        cls_token = self.cls_token.expand(b,-1,-1)
        audio_tokens = torch.cat((cls_token,audio_tokens),dim=1)

        audio_pos_ids = list(range(self.token_length_per_frame + 1))
        audio_pos_ids = torch.tensor(audio_pos_ids, dtype=torch.long, device=audio_spectrograms.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(audio_pos_ids)
        
        embeddings = audio_tokens + position_embeddings 
        #embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)



        return embeddings, position_embeddings




class Config():
    def __init__(self):
        self.void = 'void'




class OPTModel(nn.Module):

    def __init__(self, config, video_cfg, audio_cfg):
        super().__init__()

        base_cfg = Config()
        base_cfg.attention_dropout = 0.1
        base_cfg.hidden_act= "gelu"
        base_cfg.hidden_dropout= 0.1
        base_cfg.hidden_size= 768
        base_cfg.initializer_range = 0.02
        base_cfg.intermediate_size = 3072
        base_cfg.num_attention_heads = 12

        ### txt embedding and txt encoder

        has_txt_encoder = getattr(config,'has_txt_encoder',True)
        has_video_encoder = getattr(config,'has_video_encoder',True)
        has_audio_encoder = getattr(config,'has_audio_encoder',True)
        
        model_cfg_txt = copy.copy(base_cfg)
        model_cfg_txt.num_hidden_layers = config.txt_layer_num
        model_cfg_txt.vocab_size = config.vocab_size
        model_cfg_txt.max_position_embeddings = config.max_position_embeddings
        if hasattr(config,'txt_mask_prob'):
            model_cfg_txt.txt_mask_prob = config.txt_mask_prob

        self.txt_embeddings = OPTTextEmbeddings(model_cfg_txt)

        if has_txt_encoder:
            self.txt_encoder = OPTEncoder(model_cfg_txt, mode='postnorm')

        ### video embedding and video encoder
        self.video_encoder_type = config.video_encoder_type
        self.frame_num = video_cfg['sample_num']
        self.patch_size = video_cfg['patch_size']
        self.token_length_per_frame = (video_cfg['resolution'] // self.patch_size) **2
        model_cfg_video = copy.copy(base_cfg)
        model_cfg_video.num_hidden_layers = config.video_layer_num
        model_cfg_video.video_encoder_type = config.video_encoder_type

        if has_video_encoder:
            if self.video_encoder_type.startswith('vit'):
                self.video_embeddings = VitVideoEmbeddings(model_cfg_video, video_cfg)
                self.video_encoder = OPTEncoder(config, mode='prenorm')

            elif self.video_encoder_type.startswith('timesformer'):
                self.video_embeddings = TimesformerVideoEmbeddings(model_cfg_video, video_cfg)
                self.video_encoder = TimesFormerEncoder(model_cfg_video,video_cfg)
            
            elif self.video_encoder_type == 'videoswin':
                self.time_stride = config.videoswin_timestride
                self.video_encoder = SwinTransformer3D(time_stride = self.time_stride)
                self.sample_num = video_cfg['sample_num']
                self.token_length_per_frame = (video_cfg['resolution'] // video_cfg['patch_size']) **2
                self.position_embeddings = nn.Embedding(self.token_length_per_frame, config.hidden_size)
                self.frame_embedding = nn.Embedding(10,config.hidden_size)  ###assert max 10 frames
        
        ### audio embedding and audio encoder
        model_cfg_audio = copy.copy(base_cfg)
        model_cfg_audio.num_hidden_layers = config.audio_layer_num
        if has_audio_encoder:
            self.audio_embeddings = OPTAudioEmbeddings(model_cfg_audio, audio_cfg)
            self.audio_encoder = OPTEncoder(model_cfg_audio, mode='prenorm')


        ### multimodal encoder
        
        model_cfg_multimodal = copy.deepcopy(base_cfg)
        model_cfg_multimodal.num_hidden_layers = config.multimodal_layer_num
        multimodal_norm_mode = config.multimodal_norm_mode 
        self.multimodal_encoder = OPTEncoder(model_cfg_multimodal, mode= multimodal_norm_mode)


        #self.caption_encoder = self.multimodal_encoder
        self.cls_token_TV = nn.Parameter(0.02 * torch.randn(1, 1, base_cfg.hidden_size)) 
        self.txt_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, base_cfg.hidden_size)) 
        self.video_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, base_cfg.hidden_size)) 
        self.audio_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, base_cfg.hidden_size)) 
        self.reuse_embedding = config.reuse_embedding
        self.average_video  = config.average_video
        self.average_video_mode = getattr(config,'average_video_mode','space')
        self.average_audio_mode = getattr(config,'average_audio_mode','space')
        #### single modality encoder weights
        self.txt_encoder_weights = config.txt_encoder_weights
        self.video_encoder_weights = config.video_encoder_weights
        self.audio_encoder_weights = config.audio_encoder_weights

        self.audio_cfg = audio_cfg
        self.video_cfg = video_cfg


    def forward_txt_encoder(self, txt_tokens, perform_mask=False, txt_mask_indicator = None):
        attn_mask_txt = (txt_tokens != 0).long()
        txt_embeddings, txt_position_embeddings, txt_labels = self.txt_embeddings(txt_tokens, \
            perform_mask=perform_mask,txt_mask_indicator = txt_mask_indicator )
        attn_masks = attn_mask_txt.unsqueeze(1).expand(-1, txt_tokens.shape[-1], -1).clone()
        # if multimodal_uniattn:
        #     attn_masks = torch.tril(attn_masks)
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks = attn_masks.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        ##### if not add the following line, the mlm accuracy will boost a lot, which deserves futher research!!
        ### which means the txt encoder is bidirection and the multimodal encoder is uni-direction can improve the mlm
        attn_masks = (1.0 - attn_masks) * -10000.0
        txt_output = self.txt_encoder(txt_embeddings, attn_masks)
        return txt_output, txt_position_embeddings, attn_mask_txt, txt_labels




    def forward_video_encoder(self, video_pixels, perform_mask=False):
        ### b,n,c,H,W
        if self.video_encoder_type.startswith('vit'):
            video_embeddings, position_embeddings, frame_embeddings = self.video_embeddings(video_pixels) #[b, (n*f+1), c]
            video_output = self.video_encoder(video_embeddings)
            video_output = video_output.reshape(-1, self.frame_num, *video_output.shape[-2:])
            batch_size, hidden_size = video_output.shape[0], video_output.shape[-1]
            if self.video_encoder_type == 'vit_global':
                video_output = video_output[:,:,0]
                position_embeddings =  frame_embeddings
            elif self.video_encoder_type == 'vit_local':
                video_output = video_output.reshape(batch_size, -1, hidden_size)
                position_embeddings = position_embeddings.repeat(1,self.frame_num,1)
                position_embeddings = position_embeddings + frame_embeddings
            return video_output, position_embeddings

        elif self.video_encoder_type.startswith('timesformer'):
            video_embeddings, position_embeddings, video_mask_indicator = self.video_embeddings(video_pixels, perform_mask = perform_mask) #[b, (n*f+1), c]
            video_output = self.video_encoder(video_embeddings)

        elif self.video_encoder_type == 'videoswin':
            video_output = self.video_encoder(video_pixels.transpose(1,2))
            ## b,c,n,h,w
            video_output = video_output.permute(0, 2, 3, 4, 1)
            ### b,n,h,w,c
            video_output = video_output.reshape(video_output.shape[0],-1,video_output.shape[-1])
            ### b, n*h*w, c
            #video_output = torch.cat((self.video_cls_token.expand(video_output.shape[0],-1,-1),video_output),dim=1)

            sample_num = self.sample_num // self.time_stride

            video_pos_ids = list(range(self.token_length_per_frame)) * sample_num
            video_pos_ids = torch.tensor(video_pos_ids, dtype=torch.long, device=video_pixels.device).unsqueeze(0)
            position_embeddings = self.position_embeddings(video_pos_ids)

            frame_ids = [i for i in range(sample_num) for j in range(self.token_length_per_frame)]
            frame_ids = torch.tensor(frame_ids, dtype=torch.long, device=video_pixels.device).unsqueeze(0)
            position_embeddings += self.frame_embedding(frame_ids)

        return video_output, position_embeddings, video_mask_indicator
    

    def forward_audio_encoder(self, audio_spectrograms):
        audio_embeddings, position_embeddings  = self.audio_embeddings(audio_spectrograms) 
        audio_output = self.audio_encoder(audio_embeddings)
        return audio_output, position_embeddings
    

    def get_multimodal_forward_input_txt(self, txt_output, txt_position_embedding, reduction = True):
        if self.reuse_embedding:
            txt_output = txt_output + self.txt_type_embeddings + txt_position_embedding
        if reduction:
            txt_output = txt_output[:,0:1]
        return txt_output

    def get_multimodal_forward_input_video(self, video_output, video_position_embedding, video_mask_indicator, reduction = True):
        if video_mask_indicator is not None:   #### refill in the mask_token
            cls_embedding = video_output[:,0:1]
            res_embedding = video_output[:,1:]
            b,_,c = video_output.shape
            n = self.frame_num * self.token_length_per_frame 
            fillin_embedding = torch.zeros((b,n,c)).to(video_output)
            fillin_embedding[:,:] = self.video_embeddings.mask_token
            unmasked_idx = ~video_mask_indicator.bool()
            fillin_embedding[unmasked_idx] = res_embedding.reshape(-1,c)
            video_output = torch.cat((cls_embedding, fillin_embedding),dim=1)
            assert reduction == False  ### conflict 

        if self.reuse_embedding:
            video_output = video_output + self.video_type_embeddings + video_position_embedding
         
        if reduction:
            batch_size,_,hidden_size = video_output.shape
            average_video = video_output[:,1:].reshape(batch_size,self.frame_num, self.token_length_per_frame,hidden_size)
            if self.average_video_mode == 'time':
                average_video = average_video.mean(dim=1)
            elif self.average_video_mode == 'space':

                average_video = average_video.mean(dim=2)
            else:
                raise NotImplementedError
            video_output = torch.cat((video_output[:,0:1], average_video),dim=1)
        attn_masks_video = torch.ones(*video_output.shape[:2]).long().cuda()
        
        return video_output, attn_masks_video

    def get_multimodal_forward_input_audio(self, audio_output, audio_position_embedding, reduction = True):
        if self.reuse_embedding:
            audio_output = audio_output + self.audio_type_embeddings + audio_position_embedding
        
        if reduction:
            if self.average_audio_mode == 'space':
                average_audio =  audio_output[:,1:]
                average_audio = average_audio.mean(dim=1,keepdim=True)
                audio_output = torch.cat((audio_output[:,0:1], average_audio),dim=1)
            else:
                raise NotImplementedError()
        

        attn_masks_audio = torch.ones(*audio_output.shape[:2]).long().cuda()
        return audio_output, attn_masks_audio

    def forward_multimodal_encoder(self, txt_output, attn_masks_txt, video_output, attn_masks_video, audio_output=None, attn_masks_audio=None):
        if txt_output is None and audio_output is None: ### only video input
            multimodal_output = self.multimodal_encoder(video_output)

        elif audio_output is None  and  attn_masks_audio is None:  #### m2
            attn_masks_multimodal_clstoken = torch.ones(attn_masks_txt.shape[0]).to(attn_masks_txt).unsqueeze(1)
            attn_masks = torch.cat((attn_masks_multimodal_clstoken, attn_masks_txt, attn_masks_video),dim=1)
            attn_masks = attn_masks.unsqueeze(1).unsqueeze(2)
            attn_masks = attn_masks.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attn_masks = (1.0 - attn_masks) * -10000.0
            multimodal_input = torch.cat((self.cls_token_TV.expand(txt_output.shape[0],-1,-1), txt_output, video_output),dim=1)
            multimodal_output = self.multimodal_encoder(multimodal_input, attn_masks)

        else:     #### m3
            attn_masks_multimodal_clstoken = torch.ones(attn_masks_txt.shape[0]).to(attn_masks_txt).unsqueeze(1)
            attn_masks = torch.cat((attn_masks_multimodal_clstoken, attn_masks_txt, attn_masks_video,attn_masks_audio),dim=1)
            attn_masks = attn_masks.unsqueeze(1).unsqueeze(2)
            attn_masks = attn_masks.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attn_masks = (1.0 - attn_masks) * -10000.0
            multimodal_input = torch.cat((self.cls_token_TV.expand(txt_output.shape[0],-1,-1), \
                                                    txt_output, video_output, audio_output),dim=1)
            multimodal_output = self.multimodal_encoder(multimodal_input, attn_masks) 

        return multimodal_output

    def forward_caption_encoder(self, txt_tokens, attn_mask_txt, video_input, attn_masks_video, \
                                    audio_input=None, attn_masks_audio=None, perform_mask=True):

        txt_embeddings, _ , txt_labels = self.txt_embeddings(txt_tokens, perform_mask = perform_mask)
        if self.reuse_embedding:
            txt_embeddings = txt_embeddings + self.txt_type_embeddings
        
        ### m2
        if audio_input is None and  attn_masks_audio is None: 
            attn_masks = torch.cat((attn_mask_txt, attn_masks_video),dim=1)
            total_len = attn_masks.shape[1]
            txt_len = txt_tokens.shape[1]
            attn_masks = attn_masks.unsqueeze(1).expand(-1, total_len, -1).clone()
            attn_masks[:, : txt_len, : txt_len] = torch.tril(attn_masks[:, : txt_len, : txt_len])
            attn_masks[:, txt_len:, : txt_len] = 0
            attn_masks = attn_masks.unsqueeze(1)
            attn_masks = attn_masks.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attn_masks = (1.0 - attn_masks) * -10000.0
            caption_input = torch.cat((txt_embeddings, video_input),dim=1)
        ### m3
        else:
            attn_masks = torch.cat((attn_mask_txt,attn_masks_video, attn_masks_audio),dim=1)
            total_len = attn_masks.shape[1]
            txt_len = txt_tokens.shape[1]
            attn_masks = attn_masks.unsqueeze(1).expand(-1, total_len, -1).clone()
            attn_masks[:, : txt_len, : txt_len] = torch.tril(attn_masks[:, : txt_len, : txt_len])
            attn_masks[:, txt_len:, : txt_len] = 0
            attn_masks = attn_masks.unsqueeze(1)
            attn_masks = attn_masks.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attn_masks = (1.0 - attn_masks) * -10000.0
            caption_input = torch.cat((txt_embeddings, video_input, audio_input),dim=1)

        caption_output = self.multimodal_encoder(caption_input, attn_masks)
        return caption_output, txt_labels


    def forward_retrieval_encoder(self, txt_input=None, attn_masks_txt=None, video_input=None,
                                 attn_masks_video=None, audio_input=None, attn_masks_audio=None):
        
        if txt_input is not None and video_input is None and audio_input is None:
            
            attn_masks = attn_masks_txt.unsqueeze(1).expand(-1, attn_masks_txt.shape[-1], -1)
            attn_masks = attn_masks.unsqueeze(1)
            attn_masks = attn_masks.to(dtype=next(self.parameters()).dtype)  
            attn_masks = (1.0 - attn_masks) * -10000.0

            multimodal_output = self.multimodal_encoder(txt_input, attn_masks)

        elif video_input is not None and txt_input is None and audio_input is None: ###assert no mask
            multimodal_output = self.multimodal_encoder(video_input)

        elif audio_input is not None and txt_input is None and video_input is None: #### assert no mask
            multimodal_output = self.multimodal_encoder(audio_input)

        elif video_input is not None and audio_input is not None and txt_input is None:
            attn_masks_multimodal_clstoken = torch.ones(attn_masks_video.shape[0]).to(attn_masks_video).unsqueeze(1)
            attn_masks = torch.cat((attn_masks_multimodal_clstoken,  attn_masks_video,attn_masks_audio),dim=1)
            attn_masks = attn_masks.unsqueeze(1).unsqueeze(2)
            attn_masks = attn_masks.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attn_masks = (1.0 - attn_masks) * -10000.0
            multimodal_input = torch.cat((self.cls_token_TV.expand(video_input.shape[0],-1,-1), \
                                                     video_input, audio_input),dim=1)
            multimodal_output = self.multimodal_encoder(multimodal_input, attn_masks) 
        
        else:
            raise NotImplementedError
        return multimodal_output
            

        

    def initialize_weights(self):
        if self.txt_encoder_weights :
            self.initialize_txt_weights()
        if self.video_encoder_weights:
            self.initialize_video_weights()
        if self.audio_encoder_weights:
            self.initialize_audio_weights()
    
    def initialize_txt_weights(self):
        bert_weight = torch.load(self.txt_encoder_weights)
        txt_weight  = {}
        ### word_embedding_weights:
        txt_weight['txt_embeddings.word_embeddings.weight'] = bert_weight['bert.embeddings.word_embeddings.weight']
        ### position_embedding weights:
        txt_weight['txt_embeddings.position_embeddings.weight'] = bert_weight['bert.embeddings.position_embeddings.weight']

        txt_weight['txt_embeddings.layernorm.weight'] = bert_weight['bert.embeddings.LayerNorm.gamma']
        txt_weight['txt_embeddings.layernorm.bias']  = bert_weight['bert.embeddings.LayerNorm.beta']

        for  i in range(12):
            txt_weight['txt_encoder.layer.'+str(i)+'.attention.linears.0.weight'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.query.weight']
            txt_weight['txt_encoder.layer.'+str(i)+'.attention.linears.0.bias'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.query.bias']
            txt_weight['txt_encoder.layer.'+str(i)+'.attention.linears.1.weight'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.key.weight']
            txt_weight['txt_encoder.layer.'+str(i)+'.attention.linears.1.bias'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.key.bias']
            txt_weight['txt_encoder.layer.'+str(i)+'.attention.linears.2.weight'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.value.weight']
            txt_weight['txt_encoder.layer.'+str(i)+'.attention.linears.2.bias']  = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.value.bias']
            txt_weight['txt_encoder.layer.'+str(i)+'.attention.linears.3.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.attention.output.dense.weight']
            txt_weight['txt_encoder.layer.'+str(i)+'.attention.linears.3.bias'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.output.dense.bias'] 
            txt_weight['txt_encoder.layer.'+str(i)+'.ff_layer.linear1.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.intermediate.dense.weight']
            txt_weight['txt_encoder.layer.'+str(i)+'.ff_layer.linear1.bias']  = bert_weight['bert.encoder.layer.'+str(i)+'.intermediate.dense.bias']
            txt_weight['txt_encoder.layer.'+str(i)+'.ff_layer.linear2.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.output.dense.weight']
            txt_weight['txt_encoder.layer.'+str(i)+'.ff_layer.linear2.bias']  = bert_weight['bert.encoder.layer.'+str(i)+'.output.dense.bias']
            txt_weight['txt_encoder.layer.'+str(i)+'.layernorm1.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.attention.output.LayerNorm.gamma']
            txt_weight['txt_encoder.layer.'+str(i)+'.layernorm1.bias']  = bert_weight['bert.encoder.layer.'+str(i)+'.attention.output.LayerNorm.beta']
            txt_weight['txt_encoder.layer.'+str(i)+'.layernorm2.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.output.LayerNorm.gamma']
            txt_weight['txt_encoder.layer.'+str(i)+'.layernorm2.bias'] = bert_weight['bert.encoder.layer.'+str(i)+'.output.LayerNorm.beta']
        
        missing_keys, unexpected_keys = self.load_state_dict(txt_weight, strict=False)
        #LOGGER.info(f'missing_keys in txt encoder: {missing_keys}')
        LOGGER.info(f'unexpected_keys in txt encoder: {unexpected_keys}')
        del(bert_weight)
        del(txt_weight)

    def initialize_video_weights(self):
        if self.video_encoder_type.startswith('timesformer'):
            video_weight={}
            vit_weight = np.load(self.video_encoder_weights)
            video_weight['video_embeddings.cls_token']  = trans(vit_weight['cls'])
            video_weight['video_embeddings.first_conv.weight'] =  trans(vit_weight['embedding/kernel']).permute(3,2,0,1)  ### need to permute?
            video_weight['video_embeddings.first_conv.bias'] = trans(vit_weight['embedding/bias'])

            pe_weight= trans(vit_weight['Transformer/posembed_input/pos_embedding']).squeeze()
            src_len = int(math.sqrt(pe_weight.shape[0] - 1))
            tgt_len = self.video_cfg['resolution'] // self.video_cfg['patch_size']
            if src_len != tgt_len:
                LOGGER.info('interpolation for pe')
                src_weight = pe_weight[1:].reshape(src_len,src_len,-1).permute(2,0,1).unsqueeze(0)
                tgt_weight = F.interpolate(src_weight, (tgt_len,tgt_len), mode='bilinear').squeeze().permute(1,2,0)
                pe_weight = torch.cat((pe_weight[0].unsqueeze(0), tgt_weight.reshape(tgt_len**2,-1)), dim=0)
            video_weight['video_embeddings.position_embeddings.weight'] = pe_weight
            #'video_embeddings.mask_embedding.weight', 
            #'video_embeddings.layernorm.weight', 
            #'video_embeddings.layernorm.bias'

            for  i in range(12):
                video_weight['video_encoder.layer.'+str(i)+'.attention_space.linears.0.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/kernel']).reshape(768,-1).permute(1,0)
                video_weight['video_encoder.layer.'+str(i)+'.attention_space.linears.0.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/bias']).reshape(768)
                video_weight['video_encoder.layer.'+str(i)+'.attention_space.linears.1.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/kernel']).reshape(768,-1).permute(1,0)
                video_weight['video_encoder.layer.'+str(i)+'.attention_space.linears.1.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/bias']).reshape(768)
                video_weight['video_encoder.layer.'+str(i)+'.attention_space.linears.2.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/kernel']).reshape(768,-1).permute(1,0)
                video_weight['video_encoder.layer.'+str(i)+'.attention_space.linears.2.bias']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/bias']).reshape(768)
                video_weight['video_encoder.layer.'+str(i)+'.attention_space.linears.3.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/kernel']).reshape(-1,768).permute(1,0)
                video_weight['video_encoder.layer.'+str(i)+'.attention_space.linears.3.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/bias'])
                video_weight['video_encoder.layer.'+str(i)+'.attention_time.linears.0.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/kernel']).reshape(768,-1).permute(1,0).fill_(0)
                video_weight['video_encoder.layer.'+str(i)+'.attention_time.linears.0.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/bias']).reshape(768).fill_(0)
                video_weight['video_encoder.layer.'+str(i)+'.attention_time.linears.1.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/kernel']).reshape(768,-1).permute(1,0).fill_(0)
                video_weight['video_encoder.layer.'+str(i)+'.attention_time.linears.1.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/bias']).reshape(768).fill_(0)
                video_weight['video_encoder.layer.'+str(i)+'.attention_time.linears.2.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/kernel']).reshape(768,-1).permute(1,0).fill_(0)
                video_weight['video_encoder.layer.'+str(i)+'.attention_time.linears.2.bias']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/bias']).reshape(768).fill_(0)
                video_weight['video_encoder.layer.'+str(i)+'.attention_time.linears.3.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/kernel']).reshape(-1,768).permute(1,0).fill_(0)
                video_weight['video_encoder.layer.'+str(i)+'.attention_time.linears.3.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/bias']).fill_(0)
                video_weight['video_encoder.layer.'+str(i)+'.ff_layer.linear1.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_0/kernel']).permute(1,0)
                video_weight['video_encoder.layer.'+str(i)+'.ff_layer.linear1.bias']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_0/bias'])
                video_weight['video_encoder.layer.'+str(i)+'.ff_layer.linear2.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_1/kernel']).permute(1,0)
                video_weight['video_encoder.layer.'+str(i)+'.ff_layer.linear2.bias']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_1/bias'])
                video_weight['video_encoder.layer.'+str(i)+'.layernorm2.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/LayerNorm_0/scale'])
                video_weight['video_encoder.layer.'+str(i)+'.layernorm2.bias']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/LayerNorm_0/bias'])
                video_weight['video_encoder.layer.'+str(i)+'.layernorm3.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/LayerNorm_2/scale'])
                video_weight['video_encoder.layer.'+str(i)+'.layernorm3.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/LayerNorm_2/bias'])
            video_weight['video_encoder.last_layernorm.weight'] = trans(vit_weight['Transformer/encoder_norm/scale'])
            video_weight['video_encoder.last_layernorm.bias'] = trans(vit_weight['Transformer/encoder_norm/bias'])

        else:
            raise NotImplementedError

        missing_keys, unexpected_keys = self.load_state_dict(video_weight, strict=False)
        #LOGGER.info(f'missing_keys in video encoder: {missing_keys}')
        LOGGER.info(f'unexpected_keys in video encoder: {unexpected_keys}')
        del(vit_weight)
        del(video_weight)

    def initialize_audio_weights(self):
    
        vit_weight = np.load(self.audio_encoder_weights)
        audio_weight = {}
        audio_weight['audio_embeddings.cls_token']  = trans(vit_weight['cls'])
        first_conv_weight = trans(vit_weight['embedding/kernel']).permute(3,2,0,1)
        first_conv_weight = torch.mean(first_conv_weight,dim=1,keepdim=True)
        audio_weight['audio_embeddings.first_conv.weight'] =  first_conv_weight  ### need to permute?
        audio_weight['audio_embeddings.first_conv.bias'] = trans(vit_weight['embedding/bias'])
        pos_weight = trans(vit_weight['Transformer/posembed_input/pos_embedding']).squeeze()
        pos_weight_cls = pos_weight[0:1]
        pos_weight_oth = pos_weight[1:]
        if self.audio_cfg['patch_size'] == 32:
            src_patch_num = 7
        elif self.audio_cfg['patch_size'] == 16:
            src_patch_num = 14
        pos_weight_oth = pos_weight_oth.reshape(src_patch_num,src_patch_num,-1).permute(2,0,1).unsqueeze(0)
        tar_patch_num_height = self.audio_cfg['melbins'] // self.audio_cfg['patch_size']
        tar_patch_num_width = self.audio_cfg['target_length'] // self.audio_cfg['patch_size']
        pos_weight_oth = F.interpolate(pos_weight_oth, size = (tar_patch_num_height,tar_patch_num_width),mode='bilinear').squeeze().permute(1,2,0).reshape(-1,768)
        pos_weight_oth = torch.cat((pos_weight_cls,pos_weight_oth),dim=0)
        audio_weight['audio_embeddings.position_embeddings.weight'] = pos_weight_oth

        for  i in range(12):
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.0.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/kernel']).reshape(768,-1).permute(1,0)
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.0.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/query/bias']).reshape(768)
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.1.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/kernel']).reshape(768,-1).permute(1,0)
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.1.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/key/bias']).reshape(768)
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.2.weight'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/kernel']).reshape(768,-1).permute(1,0)
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.2.bias']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/value/bias']).reshape(768)
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.3.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/kernel']).reshape(-1,768).permute(1,0)
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.3.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MultiHeadDotProductAttention_1/out/bias'])
            audio_weight['audio_encoder.layer.'+str(i)+'.ff_layer.linear1.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_0/kernel']).permute(1,0)
            audio_weight['audio_encoder.layer.'+str(i)+'.ff_layer.linear1.bias']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_0/bias'])
            audio_weight['audio_encoder.layer.'+str(i)+'.ff_layer.linear2.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_1/kernel']).permute(1,0)
            audio_weight['audio_encoder.layer.'+str(i)+'.ff_layer.linear2.bias']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/MlpBlock_3/Dense_1/bias'])
            audio_weight['audio_encoder.layer.'+str(i)+'.layernorm1.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/LayerNorm_0/scale'])
            audio_weight['audio_encoder.layer.'+str(i)+'.layernorm1.bias']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/LayerNorm_0/bias'])
            audio_weight['audio_encoder.layer.'+str(i)+'.layernorm2.weight']  = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/LayerNorm_2/scale'])
            audio_weight['audio_encoder.layer.'+str(i)+'.layernorm2.bias'] = trans(vit_weight['Transformer/encoderblock_'+str(i)+'/LayerNorm_2/bias'])
        audio_weight['audio_encoder.last_layernorm.weight'] = trans(vit_weight['Transformer/encoder_norm/scale'])
        audio_weight['audio_encoder.last_layernorm.bias'] = trans(vit_weight['Transformer/encoder_norm/bias'])

        missing_keys, unexpected_keys = self.load_state_dict(audio_weight, strict=False)
        #LOGGER.info(f'missing_keys in audio encoder: {missing_keys}')
        LOGGER.info(f'unexpected_keys in audio encoder: {unexpected_keys}')
        del(vit_weight)
        del(audio_weight)


def trans(x):
    return torch.from_numpy(x)