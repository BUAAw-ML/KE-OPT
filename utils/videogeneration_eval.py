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
# import ipdb
import os
import json
from PIL import Image
import ipdb
import math
from utils.logger import LOGGER, TB_LOGGER

@torch.no_grad()
def evaluate(model, eval_loader, save_dir, global_step):
    st = time()
    LOGGER.info("start running Video generation evaluation ...")

    model.eval()

    if hvd.rank() == 0:
        pbar = tqdm(total=len(eval_loader))
    else:
        pbar = NoOp()
    
    val_loss = 0
    n_item = 0
    for batches in eval_loader:
        ids = batches['ids']
        #loss = model(batches, mode='training') 
        
        # val_loss += loss.sum().item()
        # n_item += loss.shape[0] * loss.shape[1]


        generated_frames , gt_frames= model(batches, mode='decoding')  ### batch_size X sample_num X 3 X H X W

        for i, (generated_frame, gt_frame) in enumerate(zip(generated_frames, gt_frames)):
            video_id = ids[i]
            fp = os.path.join(save_dir,'{}.jpg'.format(video_id))
            fp_gt = os.path.join(save_dir,'{}_gt.jpg'.format(video_id))
            #ipdb.set_trace()
            grid = make_grid(generated_frame)
            grid_gt = make_grid(gt_frame)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            ndarr_gt = grid_gt.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im_gt = Image.fromarray(ndarr_gt)
            im.save(fp)
            im_gt.save(fp_gt)
        pbar.update(1)
    
    # val_loss = sum(all_gather_list(val_loss))
    # n_item = sum(all_gather_list(n_item))

    # val_loss = val_loss / n_item

    # TB_LOGGER.add_scalar('val_loss', val_loss,
    #                                      global_step)
    model.train()
    pbar.close()

    tot_time = time()-st

    LOGGER.info(f"evaluation finished in {int(tot_time)} seconds")
    # LOGGER.info(f"evaluation finished in {int(tot_time)} seconds"
    #                 f"val_loss: {val_loss}")
    





@torch.no_grad()
def make_grid(
    frames,
    nrow = 1,
    padding = 2,
    normalize = True,
    value_range  = None,
    scale_each = True,
    pad_value = 0):
    

    if normalize is True:
        frames = frames.clone()  # avoid modifying frames in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in frames:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(frames, value_range)


    # make the mini-batch of images into a grid
    nmaps = frames.size(0)
    # xmaps = min(nrow, nmaps)
    # ymaps = int(math.ceil(float(nmaps) / xmaps))
    ymaps = min(nrow, nmaps)
    xmaps = int(math.ceil(float(nmaps) / ymaps))
    height, width = int(frames.size(2) + padding), int(frames.size(3) + padding)
    num_channels = frames.size(1)
    grid = frames.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # frames.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/framess.html#torch.frames.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(frames[k])
            k = k + 1
    return grid


