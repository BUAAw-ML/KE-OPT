from builtins import ValueError, isinstance
import json
import math
import os
from os.path import exists, join
from time import time, sleep
import ipdb
import torch
from torch.nn import functional as F


from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)

from utils.videocaption_eval import decode_sequence
from horovod import torch as hvd
import numpy as np
from PIL import Image
# import misc.utils as utils
# from utils.aic_evaler import AICEvaler
# from utils.coco_evaler import COCOEvaler

def validate(model, val_dataloaders, opts, global_step):
    #ipdb.set_trace()
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate on {task} task")
        assert 'Two' in task or 'Three' in task
        if 'Two' in task: 
            val_log = validate_2m(model, loader, task.split('--')[0])
        else:
            val_log = validate_3m(model, loader, task.split('--')[0])
        val_log = {f'valid_{task}/{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scaler_dict(val_log)
    model.train()

@torch.no_grad()
def validate_2m(model, val_loader, task):
    LOGGER.info("start running {} validation...".format(task))
    n_correct = 0
    n_word = 0
    n_correct_caption = 0
    n_word_caption = 0
    tot_score = 0
    n_ex = 0
    txt_feature = []
    video_feature = []
    val_log = {}
    for i, batch in enumerate(val_loader):
        evaluation_dict= model(batch, task=task, compute_loss=False)

        if 'contraTwo' in task.split('_'):
            normalized_txt = evaluation_dict['normalized_txt'] 
            normalized_video = evaluation_dict['normalized_video'] 
            txt_feature.append(normalized_txt)
            video_feature.append(normalized_video)

        if 'mlmTwo' in task.split('_'):
            prediction_scores  = evaluation_dict['prediction_scores'] 
            txt_labels = evaluation_dict['txt_labels'] 
            txt_labels = txt_labels[txt_labels != -1]
            n_correct += (prediction_scores.max(dim=-1)[1] == txt_labels).sum().item()
            n_word += txt_labels.numel()

        if 'unimlmTwo' in task.split('_'):
            prediction_scores_caption = evaluation_dict['prediction_scores_caption'] 
            txt_labels_caption = evaluation_dict['txt_labels_caption'] 
            txt_labels_caption = txt_labels_caption[txt_labels_caption != -1]
            n_correct_caption += (prediction_scores_caption.max(dim=-1)[1] == txt_labels_caption).sum().item()
            n_word_caption += txt_labels_caption.numel()
        
        if 'matchTwo' in task.split('_'):
            vtm_scores =  evaluation_dict['vtm_scores'] 
            ground_truth = evaluation_dict['ground_truth'] 
            predictions = vtm_scores.max(dim = 1 )[1]
            tot_score += (predictions.cpu().numpy() == ground_truth.cpu().numpy()).sum()
            n_ex += len(ground_truth)

        
    if 'mlmTwo' in task.split('_'):
        n_correct = sum(all_gather_list(n_correct))
        n_word = sum(all_gather_list(n_word))
        mlm_acc = n_correct / n_word
        val_log['mlm_acc'] = mlm_acc
    if 'unimlmTwo' in task.split('_'):
        n_correct_caption = sum(all_gather_list(n_correct_caption))
        n_word_caption = sum(all_gather_list(n_word_caption))
        unimlm_acc = n_correct_caption / n_word_caption     
        val_log['unimlm_acc'] = unimlm_acc
    if 'matchTwo' in task.split('_'):
        tot_score = sum(all_gather_list(tot_score))
        n_ex = sum(all_gather_list(n_ex))
        match_acc = tot_score / n_ex
        val_log['match_acc'] = match_acc
    if 'contraTwo' in task.split('_'):
        txt_feature = torch.cat(txt_feature, dim = 0)
        video_feature = torch.cat(video_feature, dim = 0)
        all_txt_feature = hvd.allgather(txt_feature)
        all_video_feature = hvd.allgather(video_feature)
        score_matrix_tv = torch.matmul(all_txt_feature, all_video_feature.permute(1,0))
        t2v_r1, v2t_r1 = compute_r1(score_matrix_tv)
        val_log['t2v_r1'] = t2v_r1*100
        #val_log['v2t_r1'] = v2t_r1*100

    LOGGER.info(val_log)

    return val_log


@torch.no_grad()
def validate_3m(model, val_loader, task):
    LOGGER.info("start running {} validation...".format(task))
    n_correct = 0
    n_correct_woaudio = 0
    n_word = 0
    n_correct_caption = 0
    n_word_caption = 0
    n_correct_caption_woaudio = 0
    n_word_caption_woaudio = 0
    txt_feature = []
    video_feature = []
    va_feature = []
    val_log = {}
    for i, batch in enumerate(val_loader):
        evaluation_dict= model(batch, task=task, compute_loss=False)

        if 'contraThree' in task.split('_'):
            feat_t = evaluation_dict['normalized_txt'] 
            feat_v = evaluation_dict['normalized_video'] 
            feat_va = evaluation_dict['normalized_va'] 
            txt_feature.append(feat_t)
            video_feature.append(feat_v)
            va_feature.append(feat_va)

        if 'mlmThree' in task.split('_'):
            prediction_scores  = evaluation_dict['prediction_scores'] 
            txt_labels = evaluation_dict['txt_labels'] 
            prediction_scores_woaudio  = evaluation_dict.get('prediction_scores_woaudio', prediction_scores)
            txt_labels = txt_labels[txt_labels != -1]
            n_correct += (prediction_scores.max(dim=-1)[1] == txt_labels).sum().item()
            n_correct_woaudio += (prediction_scores_woaudio.max(dim=-1)[1] == txt_labels).sum().item()
            n_word += txt_labels.numel()

        if 'unimlmThree' in task.split('_'):
            prediction_scores_caption = evaluation_dict['prediction_scores_caption'] 
            txt_labels_caption = evaluation_dict['txt_labels_caption'] 
            prediction_scores_caption_woaudio = evaluation_dict.get('prediction_scores_caption_two', prediction_scores_caption)
            txt_labels_caption_woaudio = evaluation_dict.get('txt_labels_caption_two',txt_labels_caption)
            txt_labels_caption = txt_labels_caption[txt_labels_caption != -1]
            txt_labels_caption_woaudio = txt_labels_caption_woaudio[txt_labels_caption_woaudio != -1]
            n_correct_caption += (prediction_scores_caption.max(dim=-1)[1] == txt_labels_caption).sum().item()
            n_word_caption += txt_labels_caption.numel()
            n_correct_caption_woaudio += (prediction_scores_caption_woaudio.max(dim=-1)[1] == txt_labels_caption_woaudio).sum().item()
            n_word_caption_woaudio += txt_labels_caption_woaudio.numel()


        
    if 'mlmThree' in task.split('_'):
        n_correct = sum(all_gather_list(n_correct))
        n_correct_woaudio = sum(all_gather_list(n_correct_woaudio))
        n_word = sum(all_gather_list(n_word))
        mlm_acc = n_correct / n_word
        mlm_acc_woaudio = n_correct_woaudio / n_word
        val_log['mlm_acc'] = mlm_acc
        val_log['mlm_acc_woaudio'] = mlm_acc_woaudio
    if 'unimlmThree' in task.split('_'):
        n_correct_caption = sum(all_gather_list(n_correct_caption))
        n_word_caption = sum(all_gather_list(n_word_caption))
        unimlm_acc = n_correct_caption / n_word_caption     

        n_correct_caption_woaudio = sum(all_gather_list(n_correct_caption_woaudio))
        n_word_caption_woaudio = sum(all_gather_list(n_word_caption_woaudio))
        unimlm_acc_woaudio = n_correct_caption_woaudio / n_word_caption_woaudio     
        val_log['unimlm_acc'] = unimlm_acc
        val_log['unimlm_acc_woaudio'] = unimlm_acc_woaudio

    if 'contraThree' in task.split('_'):
        txt_feature = torch.cat(txt_feature, dim = 0)
        video_feature = torch.cat(video_feature, dim = 0)
        va_feature = torch.cat(va_feature, dim = 0)
        all_txt_feature = hvd.allgather(txt_feature)
        all_video_feature = hvd.allgather(video_feature)
        all_va_feature = hvd.allgather(va_feature)
        score_matrix_tv = torch.matmul(all_txt_feature, all_video_feature.permute(1,0))
        score_matrix_t_va = torch.matmul(all_txt_feature, all_va_feature.permute(1,0))
        t2v_r1, v2t_r1 = compute_r1(score_matrix_tv)
        t2va_r1, va2t_r1 = compute_r1(score_matrix_t_va)
        val_log['t2v_r1'] = t2v_r1*100
        val_log['t2va_r1'] = t2va_r1*100
        #val_log['v2t_r1'] = v2t_r1*100

    LOGGER.info(val_log)

    return val_log

def compute_r1(score_matrix):
    # video retrieval

    size = len(score_matrix)
    _, rank_txt = score_matrix.topk(size, dim=1)
    gt_video = torch.arange(size).long().to(rank_txt.device).unsqueeze(1).expand_as(rank_txt)
    rank = (rank_txt == gt_video).nonzero()[:,1]
    vr_r1 = (rank < 1).sum().item() / size
    vr_r5 = (rank < 5).sum().item() / size
    vr_r10 = (rank < 10).sum().item() / size
    v_medianR = torch.median(rank) +1

    # text retrieval
 
    _, rank_video = score_matrix.topk(size, dim=0)
    gt_video = torch.arange(size).long().to(rank_txt.device).unsqueeze(0).expand_as(rank_video)
    rank = (rank_video == gt_video).nonzero()[:,0]  
    tr_r1 = (rank < 1).sum().item() / size
    tr_r5 = (rank < 5).sum().item() / size
    tr_r10 = (rank < 10).sum().item() / size
    t_medianR = torch.median(rank) +1

    return vr_r1, tr_r1
