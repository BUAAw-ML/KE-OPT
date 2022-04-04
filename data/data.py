"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Dataset interfaces
"""

import json
from toolz.sandbox import unzip
import torch
from torch.utils.data import Dataset
import horovod.torch as hvd
from torchvision.transforms.transforms import *
from torchvision import transforms
import random
from os.path import join 
from pytorch_pretrained_bert import BertTokenizer
import os
import torchaudio
from PIL import Image
from utils.logger import LOGGER
import ipdb

import string 


punctuation = string.punctuation


class TxtMapper(object):
    def __init__(self, txt_dir, max_txt_len=30, data_type= '',language = 'en'):

        self.data_type=data_type
        self.max_txt_len=max_txt_len
        self.txt_dir = txt_dir
        meta = json.load(open(join('./data/meta.json')))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']
        self.eos = 10
        
        
        
        if os.path.isfile(txt_dir):
            self.json_dict = json.load(open(txt_dir))
            if language == 'en':
                toker = './bert-base-uncased-vocab.txt'    
            elif language == 'zh':
                toker = './bert-base-chinese-vocab.txt'
            self.tokenizer = BertTokenizer.from_pretrained(toker,do_lower_case='uncased' in toker)

            self.punctuations = string.punctuation

    def __getitem__(self, id_):
        

        text = self.json_dict[id_]

        if isinstance(text,list):
            text = random.choice(text)

        text = self.clean(text) 

        txt_tokens = self.tokenize(text)

        output =self.get_padded_tokens(txt_tokens)
        
        return output

    def get_padded_tokens(self,txt_tokens):
        txt_tokens = txt_tokens[:self.max_txt_len]

        txt_tokens = [self.cls_] + txt_tokens + [self.eos] 
        
        txt_tokens = torch.tensor(txt_tokens, dtype=torch.long)

        output = torch.zeros(self.max_txt_len + 2, dtype=torch.long)
        output[:len(txt_tokens)] = txt_tokens
        return output

    def clean(self, text):
        """ lower and remove punctuations """
        text = text.lower()
        for i in self.punctuations:
            text = text.replace(i,'')
        return text

    def tokenize(self, text):
        ids = []
        for word in text.strip().split():
            ws = self.tokenizer.tokenize(word)
            if not ws:
            # some special char
                continue
            ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
        return ids






class VideoMapper(object):
    def __init__(self, video_dir, video_cfg, data_type = 'video', is_training=True):
        self.video_dir = video_dir
        self.sample_num = video_cfg['sample_num']
        self.resolution = video_cfg['resolution']
        self.patch_size = video_cfg['patch_size']
        self.mean = video_cfg['mean']
        self.std = video_cfg['std']
        self.patch_num = self.resolution // self.patch_size
        self.augmentation = 'randomresizedcrop_and_flip' if not 'aug' in video_cfg else video_cfg['aug'] 
        LOGGER.info(f'augmentation : {self.augmentation}')
        self.frame_syncaug = True
        self.datatype = data_type
        #### 'resize and crop like swinbert' or 'padding and resize like violet'
        if self.augmentation == 'resize_and_flip_and_crop':
            if is_training:
                self.transforms = transforms.Compose([Resize(self.resolution),
                                                    RandomHorizontalFlip(),
                                                    RandomCrop(self.resolution),
                                                    Normalize(self.mean,self.std)])
            else:
                self.transforms = transforms.Compose([Resize(self.resolution),
                                                    CenterCrop(self.resolution),
                                                    Normalize(self.mean,self.std)])

        elif self.augmentation == 'padding_and_resize':
            pass

        elif self.augmentation == 'randomresizedcrop_and_flip':
       
            if is_training:

                self.transforms = transforms.Compose([RandomResizedCrop(self.resolution),
                                                    RandomHorizontalFlip(),
                                                    Normalize(self.mean,self.std)])
            else:
                self.transforms = transforms.Compose([Resize(self.resolution),
                                    CenterCrop(self.resolution),
                                    Normalize(self.mean,self.std)])

        else:
            raise NotImplementedError()
        self.training = is_training
       

    def __getitem__(self, id_):
        
        if  self.datatype.startswith('video'):
            try:
                video_pixels = []
                frame_path = os.path.join(self.video_dir, id_)
                frames = os.listdir(frame_path)
                frames.sort()   ### ['img_0001.jpg','img_0002.jpg',...]
                frames_splited = self.split(frames)
                
                if self.training:
                    sample_idx = [random.choice(i) for i in frames_splited]
                else:
                    sample_idx = [i[(len(i)+1)//2-1] for i in frames_splited]


                for i in range(self.sample_num):
                    try:
                        frame = Image.open(os.path.join(frame_path,sample_idx[i]))
                        frame = transforms.ToTensor()(frame)   ## frame: 3XhXw
                        
                    except:
                        if self.sample_num==1:
                            return self.__getitem__(id_)
                        else:
                            if i>0:
                                frame = Image.open(os.path.join(frame_path,sample_idx[i-1]))
                            else:
                                frame = Image.open(os.path.join(frame_path,sample_idx[i+1]))
                            frame = transforms.ToTensor()(frame)    ## frame: 3XhXw
                            LOGGER.info('broken_img_{}'.format(sample_idx[i]))

                    if not self.frame_syncaug:
                        frame = self.transforms(frame)
                    video_pixels.append(frame.unsqueeze(0))
                
                video_pixels = torch.cat(video_pixels,dim=0)   ### nX3xHxW
                # if not self.augmentation == 'padding_and_resize':
                if self.frame_syncaug:
                    video_pixels = self.transforms(video_pixels)
                
                # else:
                #     h, w = video_pixels.shape[-2:]
                #     transform = transforms.Compose([Pad([0, (w-h)//2] if w>h else [(h-w)//2, 0]),
                #                                             Resize(self.resolution),
                #                                             Normalize(self.mean,self.std)])
                #     video_pixels = transform(video_pixels)
                return video_pixels

            except:
                return 

        elif self.datatype.startswith('image'):
            try:
                # img = Image.open(os.path.join(self.video_dir, id_))
                img = Image.open(os.path.join(self.video_dir, id_)+'.jpg')
                img = transforms.ToTensor()(img)
                #### img shape may be one-channel and 4-channel (CMYK) just return none and resample again 
                img = self.transforms(img)
                fake_video_pixels = img.unsqueeze(0).expand(self.sample_num,-1,-1,-1) ### copy img n times to make a static video
                return fake_video_pixels

            except:
                return 
        else:
            raise NotImplementedError()

    def split(self,frame_name_lists):
        if len(frame_name_lists) < self.sample_num:   ###padding with the last frame
            frame_name_lists += [frame_name_lists[-1]]*(self.sample_num - len(frame_name_lists))
        k, m = divmod(len(frame_name_lists), self.sample_num)
        return [frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(self.sample_num))]




class AudioMapper(object):
    def __init__(self, audio_dir, audio_cfg, data_type):
        self.audio_dir = audio_dir
        self.melbins = audio_cfg['melbins']
        self.target_length = audio_cfg['target_length']
        self.mean = audio_cfg['mean']
        self.std = audio_cfg['std']
        self.frame_shift = audio_cfg['frame_shift']

    
    def __getitem__(self, id_):
        wav_file = os.path.join(self.audio_dir, id_+'.wav')
        if os.path.exists(wav_file):
            try:
                waveform, sr = torchaudio.load(wav_file)
                waveform = waveform - waveform.mean()
                fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                        window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=self.frame_shift)
                #### fbank shape :(src_length,128)
                
                src_length = fbank.shape[0]
                p = self.target_length - src_length

                # cut and pad
                if p > 0:
                    m = torch.nn.ZeroPad2d((0, 0, 0, p))
                    fbank = m(fbank)
                elif p < 0:
                    #fbank = fbank[0:target_length, :]
                    fbank = fbank[(src_length//2 - self.target_length//2) : (src_length//2 + self.target_length//2)]

                #### fbank shape :(target_length,128)

                ### normalization
                fbank = (fbank - self.mean) / (self.std * 2)

                return fbank.permute(1,0)  ### 128, target_length
            
            except:
                return 

        else:
            return 'woaudio' ### current video do not have audio channel
        


if __name__ == '__main__':
    wav_file = "/raid/61_datasets/datasets/video_datasets/msrvtt_shchen/audio_22050hz/video9994.wav"
    waveform, sr = torchaudio.load(wav_file)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=64, dither=0.0, frame_shift=20)
    src_length = fbank.shape[0]
    p = 1024 - src_length
    ipdb.set_trace()
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        #fbank = fbank[0:target_length, :]
        fbank = fbank[(src_length//2 - 1024//2) : (src_length//2 + 1024//2)]


# class TxtVideoDataset(Dataset):
#     def __init__(self, ids_path, txt_mapper, video_mapper, split_id=True):
#         assert isinstance(txt_mapper, TxtMapper)
#         assert isinstance(video_mapper, VideoMapper)
#         self.txt_mapper = txt_mapper
#         self.video_mapper = video_mapper
#         self.ids = get_ids(ids_path, split_id)
#         self.idx = list(range(len(self.ids)))
#         self.dataset_name = self.video_mapper.datatype.split('_')[-1]
        
#     def __len__(self):
#         return len(self.ids)

    
#     def __getitem__(self, i):
#         id_ = self.ids[i]
#         txt_tokens = self.txt_mapper[id_]
#         video_pixels = self.video_mapper[id_]
#         if video_pixels is None: ###wrong img/video and needs to resample 
#             resample_idx = random.choice(self.idx)
#             LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong image/video, use {resample_idx} instead.')
#             return self.__getitem__(resample_idx)
#         return id_, txt_tokens, video_pixels




# def txtvideo_collate(inputs):
    
#     (ids , txt_tokens, video_pixels) = map(list, unzip(inputs))

#     txt_tokens = torch.stack(txt_tokens,dim = 0)
#     video_pixels = torch.stack(video_pixels,dim=0)
#     batch = {'ids': ids,
#              'txt_tokens': txt_tokens,
#              'video_pixels': video_pixels}


#     return batch


class TxtVideoAudioDataset(Dataset):
    def __init__(self, ids_path, txt_mapper, video_mapper, audio_mapper, split_id=True):
        assert isinstance(txt_mapper, TxtMapper)
        assert isinstance(video_mapper, VideoMapper)
        assert isinstance(audio_mapper, AudioMapper)
        self.txt_mapper = txt_mapper
        self.video_mapper = video_mapper
        self.audio_mapper = audio_mapper
        self.ids = get_ids(ids_path, split_id)
        self.idx = list(range(len(self.ids)))
        self.dataset_name = self.video_mapper.datatype.split('_')[-1]
        
    def __len__(self):
        return len(self.ids)

    
    def __getitem__(self, i):
        id_ = self.ids[i]
        txt_tokens = self.txt_mapper[id_]
        video_pixels = self.video_mapper[id_]
        audio_spectrograms = self.audio_mapper[id_]
        if video_pixels is None: ###wrong img/video and needs to resample 
            resample_idx = random.choice(self.idx)
            LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong image/video, use {resample_idx} instead.')
            return self.__getitem__(resample_idx)
        if audio_spectrograms is None: ### wrong audio and needs to resample
            resample_idx = random.choice(self.idx)
            LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong audio, use {resample_idx} instead.')
            return self.__getitem__(resample_idx)
        return id_, txt_tokens, video_pixels, audio_spectrograms

def txtvideoaudio_collate(inputs):
    
    (ids , txt_tokens, video_pixels, audio_spectrograms) = map(list, unzip(inputs))

    ids_2m = []
    txt_tokens_2m = []
    video_pixels_2m = []

    ids_3m = []
    txt_tokens_3m = []
    video_pixels_3m = []
    audio_spectrograms_3m = []

    for i in range(len(audio_spectrograms)):
        if audio_spectrograms[i] != 'woaudio':
            ids_3m.append(ids[i])
            txt_tokens_3m.append(txt_tokens[i])
            video_pixels_3m.append(video_pixels[i])
            audio_spectrograms_3m.append(audio_spectrograms[i])
        else:
            ids_2m.append(ids[i])
            txt_tokens_2m.append(txt_tokens[i])
            video_pixels_2m.append(video_pixels[i])

    if ids_3m != []:
        txt_tokens_3m = torch.stack(txt_tokens_3m, dim=0)
        video_pixels_3m = torch.stack(video_pixels_3m, dim=0)
        audio_spectrograms_3m = torch.stack(audio_spectrograms_3m, dim=0)
    if ids_2m != []:
        txt_tokens_2m = torch.stack(txt_tokens_2m, dim=0)
        video_pixels_2m = torch.stack(video_pixels_2m, dim=0)

    
    batch_2m =   {'ids': ids_2m,
             'txt_tokens': txt_tokens_2m,
             'video_pixels': video_pixels_2m}
    
    
    batch_3m =   {'ids': ids_3m,
             'txt_tokens': txt_tokens_3m,
             'video_pixels': video_pixels_3m,
             'audio_spectrograms': audio_spectrograms_3m}

    batch={'batch_2m':batch_2m,
            'batch_3m':batch_3m}

    return batch

def get_ids(ids_path,split_id=True):
    ids = []
    ids_path = json.load(open(ids_path))
    if split_id:
        for id_ in ids_path[hvd.rank()::hvd.size()]:
            ids.append(id_)
    else:
        ids = ids_path
    return ids



class MapperGroup(object):
    def __init__(self):
        self.all_mappers={}

    def set_txt_mapper(self, txt_path, max_txt_len, data_type):
        if txt_path not in self.all_mappers:
            txt_mapper = TxtMapper(txt_path, max_txt_len, data_type)
            self.all_mappers[txt_path] = txt_mapper 
        else:
            txt_mapper = self.all_mappers[txt_path]
        return txt_mapper

    def set_video_mapper(self, video_path, video_cfg, data_type, is_training=True):
        if video_path not in self.all_mappers:
            video_mapper = VideoMapper(video_path, video_cfg, data_type, is_training= is_training)
            self.all_mappers[video_path] = video_mapper 
        else:
            video_mapper = self.all_mappers[video_path]
        return video_mapper

    def set_audio_mapper(self, audio_path, audio_cfg,data_type):
        if audio_path not in self.all_mappers:
            audio_mapper = AudioMapper(audio_path, audio_cfg,data_type)
            self.all_mappers[audio_path] = audio_mapper 
        else:
            audio_mapper = self.all_mappers[audio_path]
        return audio_mapper