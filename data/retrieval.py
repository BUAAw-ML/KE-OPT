# """
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Itm dataset
# """
# from collections import defaultdict
# import copy
# import random
# import json
# import torch
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data.dataset import Dataset
# from toolz.sandbox import unzip
# from cytoolz import concat
# import numpy as np

# from .data import (TxtVideoDataset, VideoMapper, TxtMapper  )
# from .sampler import TokenBucketSampler
# from tqdm import tqdm
# from time import time 



# class SlowRetrievalEvalDataset(TxtVideoDataset):
#     def __init__(self, ids_path, txt_mapper, video_mapper, group_size=250):
#         super().__init__(ids_path, txt_mapper, video_mapper)
#         self.all_ids = json.load(open(ids_path))
#         self.group_size = group_size


#     def __getitem__(self,idx):
#         txt_id = self.ids[idx]
#         txt_tokens = self.txt_mapper[txt_id]
#         batches=[]
#         start = 0
#         while start < len(self.all_ids) - 1:
#             #video_pixels = [self.video_mapper[self.all_ids[i]] for i in range(start, start + self.group_size)]
#             #batch = self.get_batch(txt_tokens, video_pixels)
#             batch = {}
#             batch['txt_id'] = txt_id
#             batch['video_id'] = [self.all_ids[i] for i in range(start, start + self.group_size)]
#             batches.append(batch)
#             start += self.group_size
#         return batches
    
#     def get_batch(self, txt_tokens, video_pixels):
#         txt_tokens = [txt_tokens for i in range(self.group_size)]
#         txt_tokens = torch.stack(txt_tokens,dim=0)
#         video_pixels = torch.stack(video_pixels,dim=0)
#         batch = {'txt_tokens': txt_tokens,
#                 'video_pixels': video_pixels}
#         return batch


# def slowretrievaleval_collate(batches):
#     assert len(batches)==1
#     return batches[0]




