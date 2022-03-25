import json
from toolz.sandbox import unzip
import torch
from torch.utils.data import Dataset
import horovod.torch as hvd
from torchvision import transforms
import random
from os.path import join 
from utils.logger import LOGGER
import ipdb
from data import VideoMapper



class ImageClassificationDataset(Dataset):
    def __init__(self, ids_path, video_mapper, label_json, split_id = True ):
        assert isinstance(video_mapper, VideoMapper)
        self.video_mapper = video_mapper
        self.ids = get_ids(ids_path, split_id)
        self.idx = list(range(len(self.ids)))
        self.dataset_name = self.video_mapper.datatype.split('_')[-1]
        self.labels = json.load(open(label_json))
        
    def __len__(self):
        return len(self.ids)

    
    def __getitem__(self, i):
        id_ = self.ids[i]

        video_pixels = self.video_mapper[id_]

        
        if video_pixels is None: ###wrong img/video and needs to resample 
            resample_idx = random.choice(self.idx)
            LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong image/video, use {resample_idx} instead.')
            return self.__getitem__(resample_idx)

        label = torch.tensor(self.labels[id_]).long()

        return id_, video_pixels, label

def imageclassification_collate(inputs):
    
    (ids , video_pixels, labels) = map(list, unzip(inputs))
    video_pixels = torch.stack(video_pixels,dim=0)
    labels = torch.stack(labels,dim=0)
    batch = {'ids':ids,
            'video_pixels':video_pixels,
            'labels':labels
            }
    
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



