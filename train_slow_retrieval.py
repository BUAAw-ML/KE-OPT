"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

OPT finetuning for video-Text Retrieval
"""
import argparse
from multiprocessing.connection import wait
import os
from os.path import exists, join
from time import time
from cv2 import normalize

import torch
from torch._C import device
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset
from apex import amp
from horovod import torch as hvd
from tqdm import tqdm
import ipdb
import math
from data import TxtVideoDataset ,txtvideo_collate
from data.retrieval import SlowRetrievalEvalDataset, slowretrievaleval_collate
from data.vqa import MultipleChoiceVQADataset, multiplechoicevqa_collate, TxtMapperForMultipleChoiceVQA
from data import (PrefetchLoader, TxtMapper, VideoMapper)
from time import time
from model.retrieval import OPTForVideoTextRetrievalSlow
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
import torch.nn.functional as F
import numpy as np
import json
import pickle
# import ipdb


def build_dataloader(dataset, collate_fn, is_train, opts, batch_size):
    # batch_size = opts.train_batch_size if is_train else 1
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=is_train, drop_last=is_train,
                            num_workers=opts.n_workers,
                            pin_memory=opts.pin_mem, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    if hvd.rank() == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
        

    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()


### overwrite video_cfg with pretrain settings
    if opts.pretrain_dir:
        pretrain_cfg = json.load(open(os.path.join(opts.pretrain_dir,'log','hps.json')))
        ### cover model_cfg 
        for k, v in pretrain_cfg['video_cfg'].items():
            opts.video_cfg[k] = v
### overwrite video_cfg with customize settings
    if opts.sample_frame > 0:
        opts.video_cfg['sample_num'] = opts.sample_frame
    if opts.resolution > 0:
        opts.video_cfg['resolution'] = opts.resolution
    if opts.patch_size > 0:
        opts.video_cfg['patch_size'] = opts.patch_size
    if opts.drop_ratio > 0:
        opts.video_cfg['drop_ratio'] = opts.drop_ratio

    # load DBs and video dirs
    data_type = getattr(opts,'data_type','video_downstream')
    txt_mapper = TxtMapper(opts.txt_path, opts.max_txt_len, data_type)
    video_mapper_train = VideoMapper(opts.video_path, opts.video_cfg, data_type,is_training=True)
    video_mapper_test = VideoMapper(opts.video_path, opts.video_cfg, data_type,is_training=False)

    train_dataset = TxtVideoDataset(opts.train_ids_path, txt_mapper, video_mapper_train) ### when training and onev-to-manyt, a random t is choosed.
    train_dataloader = build_dataloader(train_dataset, txtvideo_collate, True, opts, batch_size=opts.train_batch_size)


    eval_retrieval = getattr(opts,'eval_retrieval',False)
    eval_multiple_choice_vqa = getattr(opts,'eval_multiple_choice_vqa',False)
    eval_match = getattr(opts,'eval_match',True)

    #assert eval_retrieval or eval_multiple_choice_vqa , 'at least for one task'
    
    if eval_retrieval or eval_match:
        test_mapper_retrieval = TxtMapper(opts.txt_path_retrievaltest, opts.max_txt_len, 'downstream')
        test_dataset_fast = TxtVideoDataset(opts.test_ids_path, test_mapper_retrieval, video_mapper_test, split_id=True)
        test_dataset_pre = TxtVideoDataset(opts.test_ids_path, test_mapper_retrieval, video_mapper_test, split_id=False)
        test_dataset = SlowRetrievalEvalDataset(opts.test_ids_path, test_mapper_retrieval, video_mapper_test)
        test_loader_fast = build_dataloader(test_dataset_fast, txtvideo_collate, False, opts, batch_size=opts.val_batch_size)
        test_loader_pre = build_dataloader(test_dataset_pre, txtvideo_collate, False, opts, batch_size=opts.val_batch_size)
        test_loader = build_dataloader(test_dataset, slowretrievaleval_collate, False, opts, batch_size = 1)


    if eval_multiple_choice_vqa:
        txt_mapper_MCQA = TxtMapperForMultipleChoiceVQA(opts.txt_path_MCtest, opts.max_txt_len)
        test_dataset_MCQA = MultipleChoiceVQADataset(opts.test_ids_path,txt_mapper_MCQA,video_mapper_test)
        test_loader_MCQA =  build_dataloader(test_dataset_MCQA, multiplechoicevqa_collate, False, opts)

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint, map_location = device)
    elif opts.pretrain_dir:
        checkpoint_dir = os.path.join(opts.pretrain_dir,'ckpt')
        checkpoint_ls = [ i for i in os.listdir(checkpoint_dir) if i.startswith('model_step')]
        checkpoint_ls.sort()
        checkpoint_name = checkpoint_ls[-1]
        checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_name), map_location = device)

        pretrain_cfg = json.load(open(os.path.join(opts.pretrain_dir,'log','hps.json')))
        ### cover model_cfg 
        for k, v in pretrain_cfg['model_cfg'].items():
            opts.model_cfg[k] = v
        ####check for position_embedding and interpolate if not equaled
        if 'opt.video_embeddings.position_embeddings.weight' in checkpoint:
            pe_weight= checkpoint['opt.video_embeddings.position_embeddings.weight']
            src_len = int(math.sqrt(pe_weight.shape[0] - 1))
            tgt_len = opts.video_cfg['resolution'] // opts.video_cfg['patch_size']
            if src_len != tgt_len:
                LOGGER.info('interpolation for pe')
                src_weight = pe_weight[1:].reshape(src_len,src_len,-1).permute(2,0,1).unsqueeze(0)
                tgt_weight = F.interpolate(src_weight, (tgt_len,tgt_len), mode='bilinear').squeeze().permute(1,2,0)
                tgt_weight = torch.cat((pe_weight[0].unsqueeze(0), tgt_weight.reshape(tgt_len**2,-1)), dim=0)
                checkpoint['opt.video_embeddings.position_embeddings.weight'] = tgt_weight
        ####check for frame_embedding and interpolate if not equaled
        if 'opt.video_embeddings.frame_embedding.weight' in checkpoint:
            src_framenum = pretrain_cfg['video_cfg']['sample_num']
            tgt_framenum = opts.video_cfg['sample_num']
            if src_framenum != tgt_framenum:
                LOGGER.info('interpolation for frame embedding')
                src_weight = checkpoint['opt.video_embeddings.frame_embedding.weight']
                valid_src_weight = src_weight[src_framenum-1:src_framenum]
                valid_tgt_weight = F.interpolate(valid_src_weight.unsqueeze(0).permute(0,2,1),tgt_framenum, mode='linear').squeeze().permute(1,0)
                src_weight[:tgt_framenum] = valid_tgt_weight
                checkpoint['opt.video_embeddings.frame_embedding.weight'] = src_weight
        LOGGER.info("Load Checkpoint {}".format(opts.checkpoint))
    else:
        checkpoint = {}

    #opts.model_cfg['video_dim'] = (opts.video_cfg['patch_size'] **2 ) * 3


    if opts.multimodal_norm_mode:
        opts.model_cfg['multimodal_norm_mode'] = opts.multimodal_norm_mode
    if opts.reuse_embedding:
        opts.model_cfg['reuse_embedding'] = opts.reuse_embedding
    if opts.video_encoder_type:
        opts.model_cfg['video_encoder_type'] = opts.video_encoder_type


    model = OPTForVideoTextRetrievalSlow.from_pretrained(
        opts.model_cfg,  checkpoint, opts.video_cfg)

    #model.init_output()  # pretrain ITM head is different from ranking head
    model.to(device)
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    model, optimizer = amp.initialize(model, optimizer,
                                      enabled=opts.fp16, opt_level='O2')

    if hvd.rank() == 0:
        save_training_meta(opts)

    global_step = 0
    LOGGER.info(f"***** Running training on {n_gpu} GPUs *****")
    LOGGER.info("  Num examples = %d", len(train_dataset) * hvd.size())
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    running_loss = RunningMeter('loss')
    model.train()

    n_epoch = 0
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()

    
    if eval_retrieval:
        test_video_rsum_best = 0
        test_video_best_iteration = 0
        test_video_rsum_fast_best = 0
        test_video_best_iteration_fast = 0
    if eval_multiple_choice_vqa:
        test_MCQA_acc_best = 0
        test_MCQA_acc_best_iteration = 0

    eval_match=False
    # only evaluation

    
       

    if eval_retrieval:
        eval_log = evaluate(model, test_loader, test_loader_pre)
        eval_log_fast = evaluate_fast(model,  test_loader_fast)
    if eval_multiple_choice_vqa:
        accuracy = evaluate_MCQA(model, test_loader_MCQA)

    if eval_match:
        match_accuracy= evaluate_match(model, test_loader_fast)

    if hvd.rank() == 0:
        if eval_retrieval:
            LOGGER.info(
                f"================ slow test iteration{global_step} ================\n"
                f"video retrieval R1: {eval_log['video_r1']*100:.2f},\n"
                f"video retrieval R5: {eval_log['video_r5']*100:.2f},\n"
                f"video retrieval R10: {eval_log['video_r10']*100:.2f},\n"
                f"video retrieval medianR: {eval_log['video_medianR']}" )
            LOGGER.info("=========================================================")

            LOGGER.info(
                f"================ fast test iteration{global_step} ================\n"
                f"video retrieval R1: {eval_log_fast['video_r1']*100:.2f},\n"
                f"video retrieval R5: {eval_log_fast['video_r5']*100:.2f},\n"
                f"video retrieval R10: {eval_log_fast['video_r10']*100:.2f},\n"
                f"video retrieval medianR: {eval_log_fast['video_medianR']}" )
            LOGGER.info("=========================================================")

        if eval_multiple_choice_vqa:
            LOGGER.info(
                f"==============    test iteration{global_step}  ==============\n"
                f"MCQA_accuracy: {accuracy*100:.2f}")
            LOGGER.info("=========================================================")

        if eval_match:
            LOGGER.info(
                f"==============    test iteration{global_step}  ==============\n"
                f"match_accuracy: {match_accuracy*100:.2f}")
            LOGGER.info("=========================================================")
    if opts.zero_shot: 
        return 

    while True:
        for step, batch in enumerate(train_dataloader):
            loss = model(batch, compute_loss=True)

            if opts.gradient_accumulation_steps > 1:
                loss = loss / opts.gradient_accumulation_steps
            delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale
                                ) as scaled_loss:
                scaled_loss.backward()
                if not delay_unscale:
                    # gather gradients from every processes
                    # do this before unscaling to make sure every process uses
                    # the same gradient scale
                    grads = [p.grad.data for p in model.parameters()
                             if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))

            running_loss(loss.item())
            if (step + 1) % opts.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, opts)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.add_scalar('loss', running_loss.val, global_step)
                if global_step % 100 == 0:
                    LOGGER.info('loss: {}'.format(running_loss.val))
                

                # update model params
                if opts.grad_norm != -1:
                    grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

                if global_step % opts.valid_steps == 0:

                    if eval_retrieval:
                        eval_log = evaluate(model, test_loader, test_loader_pre)
                        eval_log_fast = evaluate_fast(model,  test_loader_fast)
                    if eval_multiple_choice_vqa:
                        accuracy = evaluate_MCQA(model, test_loader_MCQA)
                    if eval_match:
                        match_accuracy= evaluate_match(model, test_loader_fast)

                    if hvd.rank() == 0:
                        if eval_retrieval:
                            rsum = eval_log['video_r1'] + eval_log['video_r5'] + eval_log['video_r10']
                            if rsum > test_video_rsum_best:
                                test_video_rsum_best = rsum
                                test_video_best_r1 = eval_log['video_r1']
                                test_video_best_r5 = eval_log['video_r5']
                                test_video_best_r10 = eval_log['video_r10']
                                test_video_best_medr = eval_log['video_medianR']
                                test_video_best_iteration = global_step
                            
                            rsum_fast = eval_log_fast['video_r1'] + eval_log_fast['video_r5'] + eval_log_fast['video_r10']
                            if rsum_fast > test_video_rsum_fast_best:
                                test_video_rsum_fast_best = rsum_fast
                                test_video_best_r1_fast = eval_log_fast['video_r1']
                                test_video_best_r5_fast = eval_log_fast['video_r5']
                                test_video_best_r10_fast = eval_log_fast['video_r10']
                                test_video_best_medr_fast = eval_log_fast['video_medianR']
                                test_video_best_iteration_fast = global_step

                            TB_LOGGER.log_scaler_dict({f"eval/test_{k}": v
                                                    for k, v in eval_log.items()})
                            
                            LOGGER.info(
                                f"=========== slow  test iteration{global_step} ===========================\n"
                                f"video retrieval R1: {eval_log['video_r1']*100:.2f},\n"
                                f"video retrieval R5: {eval_log['video_r5']*100:.2f},\n"
                                f"video retrieval R10: {eval_log['video_r10']*100:.2f},\n"
                                f"video retrieval medianR: {eval_log['video_medianR']}\n")
                            LOGGER.info("=========================================================")

                            LOGGER.info(
                                f"=========== fast  test iteration{global_step} ===========================\n"
                                f"video retrieval R1: {eval_log_fast['video_r1']*100:.2f},\n"
                                f"video retrieval R5: {eval_log_fast['video_r5']*100:.2f},\n"
                                f"video retrieval R10: {eval_log_fast['video_r10']*100:.2f},\n"
                                f"video retrieval medianR: {eval_log_fast['video_medianR']}\n")
                            LOGGER.info("=========================================================")

                            LOGGER.info(
                            f"======================= slow best =========================\n"
                            f"exp: {opts.output_dir},\n"
                            f"test video best_R1: {test_video_best_r1*100:.2f},\n"
                            f"test video best_R5: {test_video_best_r5*100:.2f},\n"
                            f"test video best_R10: {test_video_best_r10*100:.2f},\n"
                            f"test video best_Rsum: {test_video_rsum_best*100:.2f},\n"
                            f"test video best_medr: {test_video_best_medr},\n"
                            f"test video best_iteration: {test_video_best_iteration}")
                            LOGGER.info("=========================================================")
                            

                            LOGGER.info(
                            f"=======================+fast best+=========================\n"
                            f"exp: {opts.output_dir},\n"
                            f"test video best_R1: {test_video_best_r1_fast*100:.2f},\n"
                            f"test video best_R5: {test_video_best_r5_fast*100:.2f},\n"
                            f"test video best_R10: {test_video_best_r10_fast*100:.2f},\n"
                            f"test video best_Rsum: {test_video_rsum_fast_best*100:.2f},\n"
                            f"test video best_medr: {test_video_best_medr_fast},\n"
                            f"test video best_iteration: {test_video_best_iteration_fast}")
                            LOGGER.info("=========================================================")

                        if eval_multiple_choice_vqa:
                            if accuracy >= test_MCQA_acc_best:
                                test_MCQA_acc_best = accuracy
                                test_MCQA_acc_best_iteration = global_step

                            LOGGER.info(
                            f"==============    test_iteration{global_step}  ==============\n"
                            f"test_MCQA_acc: {accuracy*100:.2f}")
                            LOGGER.info("=========================================================")

                            TB_LOGGER.log_scaler_dict({f"eval/MCQA_test_acc": accuracy})
                            LOGGER.info(
                            f"================================================================\n"
                            f"exp: {opts.output_dir},\n"
                            f"test_MCQA_acc_best: {test_MCQA_acc_best*100:.2f},\n"
                            f"test_MCQA_acc_best_iteration: {test_MCQA_acc_best_iteration}")
                            LOGGER.info("=======================================================")

                        if eval_match:
                            LOGGER.info(
                                f"==============    test iteration{global_step}  ==============\n"
                                f"match_accuracy: {match_accuracy*100:.2f}")
                            LOGGER.info("=========================================================")
                    
                        model_saver.save(model, global_step)

                TB_LOGGER.step()

            if global_step >= opts.num_train_steps:
                break

        if global_step >= opts.num_train_steps:
            break
        n_epoch += 1
        LOGGER.info(f"finished {n_epoch} epochs")

    pbar.close()



@torch.no_grad()
def evaluate(model, test_loader, test_loader_pre):
    # """only eval on the single gpu for keeping orders"""
    # if hvd.rank()!=0:
    #     return 
    ### multiple gpu test for fast speed
    model.eval()
    
    LOGGER.info("start running slow retrieval evaluation...")

    r1 = 0
    r5 = 0
    r10 = 0
    all_item = 0
    rank_ls = []
    pre_dict={}

    if hvd.rank() == 0:
        pbar_pre = tqdm(total=len(test_loader_pre))
        pbar = tqdm(total=len(test_loader))
    else:
        pbar_pre = NoOp()
        pbar = NoOp()

    for _, batch in enumerate(test_loader_pre):
        txt_input, attn_mask_txt, video_input, attn_masks_video = model(batch, compute_loss=False, pre=True)
        ids = batch['ids']
        for i in range(len(ids)):
            pre_dict[ids[i]] =  (txt_input[i],  attn_mask_txt[i], video_input[i], attn_masks_video[i])
        
        pbar_pre.update(1)
    
    pbar_pre.close()


    for _, batches in enumerate(test_loader):

        #start = 0
        txt_id = batches[0]['txt_id']
        txt_input, attn_mask_txt = pre_dict[txt_id][:2]
        scores_ls = []


        for batch in batches:

            video_ids = batch['video_id']
            batch_size = len(video_ids)
            #ipdb.set_trace()
            txt_input_1 = txt_input.unsqueeze(0).expand(batch_size,-1,-1) 
            attn_mask_txt_1 = attn_mask_txt.unsqueeze(0).expand(batch_size,-1)
            video_input = [pre_dict[i][2].unsqueeze(0) for i in video_ids]
            attn_masks_video = [pre_dict[i][3].unsqueeze(0) for i in video_ids]
            
            video_input = torch.cat(video_input,dim=0)
            attn_masks_video = torch.cat(attn_masks_video,dim=0)

            batch['txt_input'] = txt_input_1
            batch['attn_mask_txt'] = attn_mask_txt_1
            batch['video_input'] = video_input
            batch['attn_masks_video'] = attn_masks_video

            scores = model(batch, compute_loss=False, pre=False)  ### scores shape (group_size, 1)
            for i in range(len(scores)):
                scores_ls.append([video_ids[i],scores[i].item()])

        

        scores_ls.sort(key=lambda y: -y[-1])
        scores_ls = [i[0] for i in scores_ls]
        rank = scores_ls.index(txt_id) + 1
        
        rank_ls.append(rank)
        if rank<=1:
            r1 +=1
        if rank<=5:
            r5+=1
        if rank<=10:
            r10+=1

        all_item+=1


        pbar.update(1)

    all_item = sum(all_gather_list(all_item))
    r1 = sum(all_gather_list(r1))/all_item
    r5 = sum(all_gather_list(r5))/all_item
    r10 = sum(all_gather_list(r10))/all_item
    median = [ j for i in all_gather_list(rank_ls) for j in i]
    median = np.median(np.array(median))
    
    eval_log = {
            'video_r1': r1,
            'video_r5': r5,
            'video_r10': r10,
            'video_medianR': median}
    pbar.close()
    LOGGER.info('total {} items tested'.format(all_item))
    

        
    model.train()

    return eval_log




def compute_metric(score_matrix):
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

    eval_log = {'txt_r1': tr_r1,
                'txt_r5': tr_r5,
                'txt_r10': tr_r10,
                'txt_medianR': t_medianR,
                'video_r1': vr_r1,
                'video_r5': vr_r5,
                'video_r10': vr_r10,
                'video_medianR': v_medianR}
    return eval_log

@torch.no_grad()
def evaluate_fast(model, test_loader_fast):
    model.eval()
    LOGGER.info("start running contra validation...")

    txt_feature = []
    video_pixelsure = []

    for i, batch in enumerate(test_loader_fast):
        
        pooled_txt_normalized, pooled_video_normalized = model(batch, compute_loss=False, pre=True, normalize=True)
        txt_feature.append(pooled_txt_normalized)
        video_pixelsure.append(pooled_video_normalized)

    txt_feature = torch.cat(txt_feature, dim = 0)
    video_pixelsure = torch.cat(video_pixelsure, dim = 0)
    all_txt_feature = hvd.allgather(txt_feature)
    all_video_pixelsure = hvd.allgather(video_pixelsure)

    if hvd.rank()==0:
        score_matrix_tv = torch.matmul(all_txt_feature, all_video_pixelsure.permute(1,0))
        eval_log = compute_metric(score_matrix_tv)
    else:
        eval_log = None
    model.train()
    return eval_log

@torch.no_grad()
def evaluate_match(model, test_loader_fast):
    model.eval()
    correct = 0
    total = 0
    
    
    for i, batch in tqdm(enumerate(test_loader_fast)):
    
        scores, groundtruth = model(batch, compute_loss=True, evaluate_match=True)  ### scores shape (batch_size, 2)
        _, pred = scores.max(dim=1)
        correct +=(pred == groundtruth).sum().item()
        total += len(groundtruth)

    all_correct = sum(all_gather_list(correct))
    all_total = sum(all_gather_list(total))
    acc = all_correct/all_total
    model.train()

    return acc





@torch.no_grad()
def evaluate_MCQA(model, test_loader_MCQA):
    model.eval()
    LOGGER.info("start running MCQA validation...")

    predicted_ls = []
    answers_ls = []



    for i, batch in enumerate(test_loader_MCQA):
        scores = model(batch, compute_loss=False)
        scores = scores[:,1]
        answers = batch['answers']
        scores = scores.reshape(len(answers),-1)
        choices_num = scores.shape[1]
        predicted = scores.max(dim=1)[1].cpu().numpy().tolist()
        predicted_ls+=predicted
        answers_ls+=answers
    
    predicted_ls = [ i  for ls in all_gather_list(predicted_ls) for i in ls ]
    answers_ls = [ i  for ls in all_gather_list(answers_ls) for i in ls ]


    if hvd.rank()==0:
        total_num = len(predicted_ls)
        assert len(answers_ls) == total_num
        LOGGER.info('total {} questions has been tested'.format(total_num))
        LOGGER.info('choices_num per question(video): {}'.format(choices_num))
        accurate_num = sum([predicted_ls[i] == answers_ls[i] for i in range(total_num)])
        accuracy = accurate_num / total_num
    else:
        accuracy = None
    model.train()
    return accuracy





def str2bool(b):
    if b.lower() in ["false"]:
        return False
    elif b.lower() in ["true"]:
        return True
    else:
        raise Exception("Invalid Bool Value")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained MLM")

    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model "
                             "checkpoints will be written.")

    parser.add_argument("--video_path", default=None, type=str,
                        help="The output directory where the model "
                             "checkpoints will be written.")
    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--max_video_len', type=int, default=30,
                        help='max number of tokens in text (BERT BPE)')


    # training parameters
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Total batch size for training. "
                             "(batch by examples)")
    parser.add_argument("--negative_size", default=1, type=int,
                        help="Number of negative samples per positive sample")
    parser.add_argument("--inf_minibatch_size", default=400, type=int,
                        help="batch size for running inference. "
                             "(used for validation, and evaluation)")

    parser.add_argument("--margin", default=0.2, type=float,
                        help="margin of ranking loss")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=0.25, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=4000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--full_val', action='store_true',
                        help="Always run full evaluation during training")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    parser.add_argument('--zero_shot', action='store_true',
                        help="Always run full evaluation during training")




    parser.add_argument('--sample_frame', type=int, default=0,
                        help="random seed for initialization")

    
    
    parser.add_argument('--resolution', type=int, default=0,
                        help="random seed for initialization")


    parser.add_argument('--patch_size', type=int, default=0,
                        help="random seed for initialization")

    parser.add_argument('--drop_ratio', type=float, default=0.,
                        help="random seed for initialization")

    parser.add_argument('--video_encoder_type', type=str, default=None,
                        help="random seed for initialization")

    parser.add_argument('--videoswin_timestride', type=int, default=1,
                        help="random seed for initialization")
    parser.add_argument('--eval_retrieval', type=str2bool, default=True,
                        help="random seed for initialization")
    parser.add_argument('--match_mode', type=str, default='pair_simple',
                        help="random seed for initialization")
    parser.add_argument('--multimodal_layernum', type=int, default=2,
                        help="random seed for initialization")
    parser.add_argument("--audio_path", default='', type=str,
                        help="The output directory where the model "
                             "checkpoints will be written.")
    parser.add_argument('--reuse_embedding', type=str2bool, default=True,
                        help="random seed for initialization")
    parser.add_argument('--average_video', type=str2bool, default=None,
                        help="random seed for initialization")
    parser.add_argument('--pretrain_dir', type=str, default='',
                        help="random seed for initialization")
    parser.add_argument('--multimodal_norm_mode', type=str, default=None,
                        help="random seed for initialization")
    parser.add_argument('--use_audio', type=str2bool, default=True,
                        help="random seed for initialization")
    args = parse_with_config(parser)

    
    # if exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not "
    #                      "empty.".format(args.output_dir))

    # options safe guard


    main(args)
