import argparse
from collections import defaultdict
import json
import math
import os
from os.path import exists, join
from time import time, sleep
import lmdb

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
import random
from utils.distributed import all_gather_list
from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from horovod import torch as hvd

from data import  PrefetchLoader
from data.data import TxtVideoAudioDataset, txtvideoaudio_collate
import ipdb



def build_dataloader(dataset, collate_fn, is_train, opts, batch_size):
    loader = DataLoader(dataset, batch_size = batch_size,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn, shuffle = is_train)

    return loader


def build_dataset(ids_path, txt_mapper, video_mapper, audio_mapper):
    dataset = TxtVideoAudioDataset(ids_path, txt_mapper, video_mapper, audio_mapper)
    collate_fn = txtvideoaudio_collate 
    return dataset, collate_fn


def create_train_dataloaders(data_cfg, opts, mapper_group):
    data_cfg = data_cfg['train']
    dataloaders = {}
    for d_cfg in data_cfg:
        concate_name = ''
        dataset_ls = []
        for dset in d_cfg['datasets']:
            name = dset['name']
            concate_name = concate_name + name if concate_name == '' else concate_name + '_' + name
            data_type = dset['datatype'] + '_' + name
            ids_path = dset['ids_path']
            
            video_mapper = mapper_group.set_video_mapper(dset['video'], opts.video_cfg, data_type)
            txt_mapper = mapper_group.set_txt_mapper(dset['txt'], opts.max_txt_len, data_type) 
            audio_path = dset.get('audio','')
            audio_mapper = mapper_group.set_audio_mapper(audio_path, opts.audio_cfg, data_type)

            dataset, collate_fn = build_dataset(ids_path, txt_mapper, video_mapper, audio_mapper)
            LOGGER.info("Create Dataset {} Success".format(name))
            dataset_ls.append(dataset)
        dataset = ConcatDataset(dataset_ls)
        LOGGER.info("Create Dataset {} Success".format(concate_name))
        ratio = d_cfg['mix_ratio']
        task = d_cfg['task']
        batch_size = d_cfg['batch_size']
        loader = build_dataloader(dataset, collate_fn, True, opts, batch_size)
        task_name = f'{task}--{concate_name}'
        dataloaders[task_name] = (loader, ratio)
    
    return dataloaders, mapper_group


def create_val_dataloaders(data_cfg, opts, mapper_group):
    data_cfg = data_cfg['val']
    dataloaders = {}
    for d_cfg in data_cfg:
        name = d_cfg['name']
        data_type = d_cfg['datatype']
        ids_path = d_cfg['ids_path']
        video_mapper = mapper_group.set_video_mapper(d_cfg['video'], opts.video_cfg, data_type)
        txt_mapper = mapper_group.set_txt_mapper(d_cfg['txt'], opts.max_txt_len, data_type)
        audio_path = d_cfg.get('audio','')
        audio_mapper = mapper_group.set_audio_mapper(audio_path, opts.audio_cfg, data_type)
        dataset, collate_fn = build_dataset(ids_path, txt_mapper, video_mapper, audio_mapper)
        LOGGER.info("Create Dataset {} Success".format(name))
        task = d_cfg['task']
        batch_size = d_cfg['batch_size']
        loader = build_dataloader(dataset, collate_fn, False, opts, batch_size)
        task_name = f'{task}--{name}'
        dataloaders[task_name] = PrefetchLoader(loader)
    return dataloaders, mapper_group


