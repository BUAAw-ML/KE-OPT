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

# from cococaption.pycocotools.coco import COCO
# from cococaption.pycocoevalcap.eval import COCOEvalCap

def decode_sequence(i2v, seq, EOS_token):
    N, T = seq.size()
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t].item()
            if ix == EOS_token:
                break
            words.append(i2v[ix])
        sent = ' '.join(words)
        sents.append(sent.replace(' ##', ''))
    return sents


@torch.no_grad()
def evaluate(model, eval_loader , split, opts):
    st = time()
    LOGGER.info("start running Video Caption evaluation ...")
    vocab = [line.strip() for line in open(opts.toker)]
    i2v = {i:vocab[i]  for i in range(len(vocab))}
    EOS_token = opts.model_cfg['EOS_token']

    model.eval()
    results=[]
    if hvd.rank() == 0:
        pbar = tqdm(total=len(eval_loader))
    else:
        pbar = NoOp()
    #total_loss = 0 
    #total_item = 0
    for batches in eval_loader:
        ids = batches['ids']
        #loss = model(batches, mode='training')
        #total_loss += loss.sum().item()
        #total_item += loss.size(0)
        sents = model(batches, mode='decoding')
        sents = decode_sequence(i2v, sents.data, EOS_token)

        for i, sent in enumerate(sents):
            result = {'video_id':ids[i], 'caption': sent}
            results.append(result)

        pbar.update(1)
    
    model.train()
    pbar.close()

    #total_loss = sum(all_gather_list(total_loss))
    #total_item = sum(all_gather_list(total_item))
    all_results = [i for results in all_gather_list(results)  for i in results]
    
    #total_loss = total_loss / total_item
    if hvd.rank() != 0:
        return {},{}
    
    eval_log = compute_metric(all_results, split, opts)  
    #eval_log['loss'] = total_loss
    tot_time = time()-st
    LOGGER.info(f"evaluation finished in {int(tot_time)} seconds")
    return eval_log, all_results






def compute_metric(results, split, opts):
    if split =='val':
        annfile = opts.val_annfile
    elif split =='test':      
        annfile = opts.test_annfile
    else:       
        raise ValueError
    if not annfile:
        return 
    coco = COCO(annfile)
    cocoRes = coco.loadRes(results)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    return cocoEval.eval



