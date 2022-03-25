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
from torchvision.transforms import *
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
            val_log = validate_2m(model, loader, task.split('--')[0], opts, global_step)
        else:
            val_log = validate_3m(model, loader, task.split('--')[0], opts, global_step)
        val_log = {f'valid_{task}/{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scaler_dict(val_log)
    model.train()

@torch.no_grad()
def validate_2m(model, val_loader, task,opts, global_step):
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
def validate_3m(model, val_loader, task, opts, global_step):
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
    ### mvm
    mvm_raw_pixels_regression_loss = []
    mvm_feat_regression_loss = []
    n_correct_patches = 0
    n_patches = 0
    ###
    tot_score_three = 0
    tot_score_two = 0
    n_ex = 0

    visual_vqvae_loss = []
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
            prediction_scores_woaudio  = evaluation_dict['prediction_scores_woaudio']
            txt_labels = txt_labels[txt_labels != -1]
            n_correct += (prediction_scores.max(dim=-1)[1] == txt_labels).sum().item()
            n_correct_woaudio += (prediction_scores_woaudio.max(dim=-1)[1] == txt_labels).sum().item()
            n_word += txt_labels.numel()

        if 'unimlmThree' in task.split('_'):
            prediction_scores_caption = evaluation_dict['prediction_scores_caption'] 
            txt_labels_caption = evaluation_dict['txt_labels_caption'] 
            prediction_scores_caption_woaudio = evaluation_dict['prediction_scores_caption_two']
            txt_labels_caption_woaudio = evaluation_dict['txt_labels_caption_two']
            txt_labels_caption = txt_labels_caption[txt_labels_caption != -1]
            txt_labels_caption_woaudio = txt_labels_caption_woaudio[txt_labels_caption_woaudio != -1]
            n_correct_caption += (prediction_scores_caption.max(dim=-1)[1] == txt_labels_caption).sum().item()
            n_word_caption += txt_labels_caption.numel()
            n_correct_caption_woaudio += (prediction_scores_caption_woaudio.max(dim=-1)[1] == txt_labels_caption_woaudio).sum().item()
            n_word_caption_woaudio += txt_labels_caption_woaudio.numel()


        if 'matchThree' in task.split('_'):
            scores_two =  evaluation_dict['scores_two'] 
            scores_three =  evaluation_dict['scores_three'] 
            ground_truth = evaluation_dict['ground_truth'] 
            predictions_two = scores_two.max(dim = 1 )[1]
            tot_score_two += (predictions_two.cpu().numpy() == ground_truth.cpu().numpy()).sum()
            predictions_three = scores_three.max(dim = 1 )[1]
            tot_score_three += (predictions_three.cpu().numpy() == ground_truth.cpu().numpy()).sum()
            n_ex+= len(ground_truth)


        if 'mvmThree' in task.split('_'):
            if model.mvm_target == 'raw_pixels_regression':
                mvm_raw_pixels_regression_loss.append(evaluation_dict['mvm_raw_pixels_regression_loss'].item())
                video_mask_indicator = evaluation_dict['mvm_raw_pixels_regression_mask_indicator']
                batch_size = video_mask_indicator.shape[0]
                video_mask_indicator = video_mask_indicator[0]
                video_predictions = evaluation_dict['mvm_raw_pixels_regression_predictions']
                hidden_size = video_predictions.shape[-1]
                video_predictions = video_predictions.reshape(batch_size,-1,hidden_size)[0]
                id = evaluation_dict['mvm_raw_pixels_regression_ids'][0] 
                save_dir = os.path.join(opts.output_dir, 'pixel_regresion_result',f'step_{global_step}')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                                    
                video_pixels = evaluation_dict['mvm_raw_pixels_regression_gt'][0][0] #### 3,h,w
                _,h,w = video_pixels.shape
                raw_img = video_pixels * torch.tensor([0.229, 0.224, 0.225]).to(video_pixels).view(-1,1,1) + \
                                             torch.tensor([0.485, 0.456, 0.406]).to(video_pixels).view(-1,1,1)

                masked_img = raw_img.clone()
                p_num = int(math.sqrt(video_mask_indicator.shape[0]))
                
                masked_img = masked_img.permute(1,2,0).reshape(p_num, h//p_num, p_num, w//p_num, 3)
                masked_img = masked_img.permute(0,2,1,3,4).reshape(p_num**2,-1)
                masked_img[video_mask_indicator.bool()] = 0
                masked_img = masked_img.reshape(p_num, p_num, h//p_num, w//p_num, 3).permute(4,0,2,1,3).reshape(3,h,w)

                predicted_img = video_pixels.clone()
                predicted_img = predicted_img.permute(1,2,0).reshape(p_num, h//p_num, p_num, w//p_num, 3)
                predicted_img = predicted_img.permute(0,2,1,3,4).reshape(p_num**2,-1)
                predicted_img[video_mask_indicator.bool()] = video_predictions
                predicted_img = predicted_img.reshape(p_num, p_num, h//p_num, w//p_num, 3).permute(4,0,2,1,3).reshape(3,h,w)
                predicted_img = predicted_img * torch.tensor([0.229, 0.224, 0.225]).to(video_pixels).view(-1,1,1) + \
                                             torch.tensor([0.485, 0.456, 0.406]).to(video_pixels).view(-1,1,1)

                total_img = torch.cat((raw_img,masked_img,predicted_img),dim = -1)
                total_img = ToPILImage()(total_img)
                total_img.save(os.path.join(save_dir,f'{id}.jpg'))

            elif model.mvm_target == 'raw_pixels_classification':
                video_predictions = evaluation_dict['mvm_raw_pixels_classification_logits'] 
                video_target =     evaluation_dict['mvm_raw_pixels_classification_targets']

                n_correct_patches +=  (video_predictions.max(dim=-1)[1] == video_target).sum().item()
                n_patches += video_target.numel()
            elif model.mvm_target in ['feat_regression','feat_regression_clip']:
                mvm_feat_regression_loss.append(evaluation_dict['mvm_feat_regression_loss'].item())

            elif model.mvm_target == 'feat_classification':
                video_predictions = evaluation_dict['mvm_feat_classification_logits'] 
                video_target =     evaluation_dict['mvm_feat_classification_targets']
                n_correct_patches +=  (video_predictions.max(dim=-1)[1] == video_target).sum().item()
                n_patches += video_target.numel()

        # if 'visualvqvae' in task.split('_'):
        #     save_dir = os.path.join(opts.output_dir, 'visualvqvae_result',f'step_{global_step}')
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     visual_vqvae_loss.append(evaluation_dict['visual_vqvae_loss'].item())
        #     reconstruction_results = evaluation_dict['visual_vqvae_reconstruction_results'][0] ## n,c

        #     video_pixels = batch['batch_3m']['video_pixels'][0][0]  ### 3,h,w
        #     _,h,w = video_pixels.shape
        #     raw_img = video_pixels * torch.tensor([0.229, 0.224, 0.225]).to(video_pixels).view(-1,1,1) + \
        #                         torch.tensor([0.485, 0.456, 0.406]).to(video_pixels).view(-1,1,1)
        #     p_num = int(math.sqrt(reconstruction_results.shape[0]))
        #     reconstruction_results = reconstruction_results.reshape(p_num, p_num, h//p_num, w//p_num, 3).permute(4,0,2,1,3).reshape(3,h,w)
        #     reconstruction_results = reconstruction_results * torch.tensor([0.229, 0.224, 0.225]).to(reconstruction_results).view(-1,1,1) + \
        #                         torch.tensor([0.485, 0.456, 0.406]).to(reconstruction_results).view(-1,1,1)
        #     total_img = torch.cat((raw_img,reconstruction_results),dim=-1)
        #     total_img = ToPILImage()(total_img)
        #     id = batch['batch_3m']['ids'][0]
        #     total_img.save(os.path.join(save_dir,f'{id}.jpg'))

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
    
    if 'mvmThree' in task.split('_'):
        if model.mvm_target == 'raw_pixels_regression':
            mvm_raw_pixels_regression_loss = np.mean([ j  for i in all_gather_list(mvm_raw_pixels_regression_loss) for j in i ])
            val_log['mvm_raw_pixels_regression_loss'] = mvm_raw_pixels_regression_loss

        elif model.mvm_target in ['raw_pixels_classification','feat_classification']:
            n_correct_patches =  sum(all_gather_list(n_correct_patches))
            n_patches = sum(all_gather_list(n_patches))
            mvm_acc = n_correct_patches / n_patches
            val_log['mvm_acc'] = mvm_acc

        elif model.mvm_target in ['feat_regression','feat_regression_clip']:
            mvm_feat_regression_loss = np.mean([ j  for i in all_gather_list(mvm_feat_regression_loss) for j in i ])
            val_log['mvm_feat_regression_loss'] = mvm_feat_regression_loss

    if 'matchThree' in task.split('_'):
        tot_score_two = sum(all_gather_list(tot_score_two))
        tot_score_three = sum(all_gather_list(tot_score_three))
        n_ex = sum(all_gather_list(n_ex))
        match_acc_two = tot_score_two / n_ex
        match_acc_three = tot_score_three / n_ex
        val_log['match_acc_two'] = match_acc_two
        val_log['match_acc_three'] = match_acc_three
    # if 'visualvqvae' in task.split('_'):
    #     visual_vqvae_loss = np.mean([ j  for i in all_gather_list(visual_vqvae_loss) for j in i ])
    #     val_log['visual_vqvae_loss'] = visual_vqvae_loss


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
