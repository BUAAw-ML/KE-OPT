"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

video Text Retrieval evaluation helper
"""
from time import time

import torch
from horovod import torch as hvd
from tqdm import tqdm

from .logger import LOGGER
from .misc import NoOp
from .distributed import all_gather_list
import ipdb
import os
import json

# from cococaption.pycocotools.coco import COCO
# from cococaption.pycocoevalcap.eval import COCOEvalCap




