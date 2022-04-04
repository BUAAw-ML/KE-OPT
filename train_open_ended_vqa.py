"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

OPT finetuning for video-Text Retrieval
"""
import argparse
import os
from os.path import join
from time import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from apex import amp
from horovod import torch as hvd
from tqdm import tqdm

from data import (PrefetchLoader, )
from data.data import VideoMapper,AudioMapper
from data.vqa import OpenEndedVQADataset, openendedvqa_collate, TxtMapperForOpenEndedVQA

from model.vqa import OPTForOpenEndedVQA
from optim import get_lr_sched
from optim.misc import build_optimizer_for_VQA

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_random_seed


import json
import ipdb
import math
import torch.nn.functional as F
from collections import defaultdict

def build_dataloader(dataset, collate_fn, is_train, opts):
    batch_size = opts.train_batch_size if is_train else opts.val_batch_size
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
        save_training_meta(opts)  ###saving later
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
        os.makedirs(join(opts.output_dir, 'results_test'),exist_ok=True)

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
        for k, v in pretrain_cfg['audio_cfg'].items():
            opts.audio_cfg[k] = v
### overwrite video_cfg with customize settings
    if opts.sample_frame > 0:
        opts.video_cfg['sample_num'] = opts.sample_frame
    if opts.resolution > 0:
        opts.video_cfg['resolution'] = opts.resolution
    if opts.patch_size > 0:
        opts.video_cfg['patch_size'] = opts.patch_size
    if opts.drop_ratio > 0:
        opts.video_cfg['drop_ratio'] = opts.drop_ratio

    if opts.audio_melbins is not None:
        opts.audio_cfg['melbins'] = opts.audio_melbins
    if opts.audio_target_length is not None:
        opts.audio_cfg['target_length'] = opts.audio_target_length
    if opts.audio_patch_size is not None:
        opts.audio_cfg['patch_size'] = opts.audio_patch_size
    if opts.audio_frame_shift is not None:
        opts.audio_cfg['frame_shift'] = opts.audio_frame_shift

    if not opts.use_audio:
        opts.audio_path = '' #### delete audio path for 2m exp
 
    data_type = getattr(opts,'data_type','video_downstream')
    txt_mapper = TxtMapperForOpenEndedVQA(opts.txt_path, opts.answer_candidate_path, opts.max_txt_len)
    video_mapper_train = VideoMapper(opts.video_path, opts.video_cfg, data_type,is_training=True)
    video_mapper_test = VideoMapper(opts.video_path, opts.video_cfg, data_type,is_training=False)
    audio_mapper = AudioMapper(opts.audio_path, opts.audio_cfg, data_type)

    train_dataset = OpenEndedVQADataset(opts.train_ids_path, txt_mapper, video_mapper_train, audio_mapper)

    train_dataloader = build_dataloader(train_dataset, openendedvqa_collate, True, opts)
     

    test_dataset = OpenEndedVQADataset(opts.test_ids_path, txt_mapper, video_mapper_test, audio_mapper, training=False)
    test_loader = build_dataloader(test_dataset, openendedvqa_collate, False, opts)

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

    
    opts.model_cfg['answer_num'] = len(json.load(open(opts.answer_candidate_path)))
    if opts.multimodal_norm_mode:
        opts.model_cfg['multimodal_norm_mode'] = opts.multimodal_norm_mode
    if opts.reuse_embedding:
        opts.model_cfg['reuse_embedding'] = opts.reuse_embedding
    if opts.video_encoder_type:
        opts.model_cfg['video_encoder_type'] = opts.video_encoder_type
    if opts.strengthen_two is not None:
        opts.model_cfg['strengthen_two'] = opts.strengthen_two
    if opts.woaudioweight:
        opts.model_cfg['audio_encoder_weights'] = ''
    if opts.average_video_mode is not None:
        opts.model_cfg['average_video_mode'] = opts.average_video_mode
    if opts.average_audio_mode is not None:
        opts.model_cfg['average_audio_mode'] = opts.average_audio_mode
    if opts.audio_encoder_weights is not None :
        opts.model_cfg['audio_encoder_weights'] = opts.audio_encoder_weights



    model = OPTForOpenEndedVQA.from_pretrained(
        opts.model_cfg,  checkpoint, opts.video_cfg, opts.audio_cfg)

    model.to(device)
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    #set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer_for_VQA(model, opts)
    model, optimizer = amp.initialize(model, optimizer,
                                      enabled=opts.fp16, opt_level='O2')


    if hvd.rank() == 0:
        save_training_meta(opts) #### saving again for overwrited opts

    global_step = 0
    LOGGER.info(f"***** Running training on {n_gpu} GPUs *****")
    LOGGER.info("  Num examples = %d", len(train_dataset) * hvd.size())
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    running_loss = RunningMeter('loss')
    model.train()

    loss_moving_averagetors ={}
    metric_logger_dict = defaultdict(dict)
    n_epoch = 0
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()



    # only evaluation

    eval_log = evaluate(model, test_loader , opts)
    if hvd.rank() == 0:
        for eval_name, metric in eval_log.items():
            TB_LOGGER.log_scaler_dict({f"eval/{eval_name}/test_{k}": v
                                for k, v in metric.items()})
            LOGGER.info(f"====-evaluation--{eval_name}=====step {global_step}--==========\n")
            LOGGER.info(metric)
    if opts.zero_shot:
        return 


    while True:
        
        for step, batch in enumerate(train_dataloader):

            loss_dict = model(batch, compute_loss=True)
            
            sample_num_2m = len(batch['batch_2m']['ids'])
            sample_num_3m = len(batch['batch_3m']['ids'])

            if sample_num_3m > 0:
                loss_3m = loss_dict['vqa_loss_3m']
            else:
                loss_3m = 0
            
            if sample_num_2m > 0:
                loss_2m = loss_dict['vqa_loss_2m']
            else:
                loss_2m = 0
            
            loss = sample_num_3m/(sample_num_3m + sample_num_2m) * loss_3m + \
                    sample_num_2m/(sample_num_3m + sample_num_2m) * loss_2m
            
            loss_dict['total_loss'] = loss.item()

            delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale
                                ) as scaled_loss:
                scaled_loss.backward()
                # print(list(model.parameters())[2].grad)
                
                if not delay_unscale:
                    # gather gradients from every processes
                    # do this before unscaling to make sure every process uses
                    # the same gradient scale
                    grads = [p.grad.data for p in model.parameters()
                             if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))

            for k in loss_dict.keys():
                if not k in loss_moving_averagetors:
                ### first time initialize 
                    loss_moving_averagetors[f'loss/{k}'] = RunningMeter()
            
            for k,v in loss_dict.items():
                loss_moving_averagetors[f'loss/{k}'](v)

            if (step + 1) % opts.gradient_accumulation_steps == 0:

                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, opts)
                for idx,param_group in enumerate(optimizer.param_groups):
                    if idx<2:
                        param_group['lr'] = lr_this_step
                    else:
                        param_group['lr'] = lr_this_step*5 ### vqa_head bigger lr

                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.log_scaler_dict({name: averagetor.val
                                        for name, averagetor in loss_moving_averagetors.items()
                                        if averagetor.val is not None})

                if global_step % 100 == 0:    
                    LOGGER.info({name : averagetor.val for name, averagetor in loss_moving_averagetors.items()})                                   
                   

                # update model params
                if opts.grad_norm != -1:
                    grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)



                if global_step % opts.valid_steps == 0:

                    eval_log = evaluate(model, test_loader,  opts)

                    if hvd.rank() == 0:

                        for eval_name, metric in eval_log.items():
                            metric_logger_dict[eval_name][str(global_step)] = metric
                            if ('best_step' not in metric_logger_dict[eval_name]) or \
                                    (metric['accuracy'] >= metric_logger_dict[eval_name]['best_value']):
                                metric_logger_dict[eval_name]['best_step'] = global_step
                                metric_logger_dict[eval_name]['best_value'] = metric['accuracy']

                            best_step = metric_logger_dict[eval_name]['best_step']
                            TB_LOGGER.log_scaler_dict({f"eval/{eval_name}/test_{k}": v
                                                for k, v in metric.items()})
                            LOGGER.info(f"====-evaluation--{eval_name}=====step {global_step}--==========\n")
                            LOGGER.info(metric)
                            LOGGER.info(f"======evaluation--{eval_name}====history best step: {best_step}==\n")
                            LOGGER.info(metric_logger_dict[eval_name][str(best_step)])
            
                        
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
def evaluate(model, eval_loader , opts):
    st = time()
    LOGGER.info("start running open-ended vqa evaluation ...")
    answer_candidate = json.load(open(opts.answer_candidate_path))
    answer_candidate = {v:k for k,v in answer_candidate.items()}
    model.eval()

    if hvd.rank() == 0:
        pbar = tqdm(total=len(eval_loader))
    else:
        pbar = NoOp()

    predicted_answers_2m=[]
    predicted_answers_3m=[]
    groundtruth_answers=[]
    for batch in eval_loader:
        
        ids_2m = batch['batch_2m']['ids']
        ids_3m = batch['batch_3m']['ids']
        sample_num_2m = len(ids_2m)
        sample_num_3m = len(ids_3m)

        evaluation_dict = model(batch, compute_loss=False)

        if sample_num_2m > 0:
            
            indexes = evaluation_dict['logits_2m'].max(dim=1)[1].cpu().numpy().tolist()
            answers = [answer_candidate[i] for i in indexes]
            predicted_answers_2m += answers
            predicted_answers_3m += answers
            groundtruth_answers += batch['batch_2m']['answers']

        if sample_num_3m > 0:
            
            indexes = evaluation_dict['logits_3m'].max(dim=1)[1].cpu().numpy().tolist()
            answers = [answer_candidate[i] for i in indexes]
            predicted_answers_3m += answers

            indexes_woaudio = evaluation_dict['logits_3m_woaudio'].max(dim=1)[1].cpu().numpy().tolist()
            answers_woaudio = [answer_candidate[i] for i in indexes_woaudio]
            predicted_answers_2m += answers_woaudio 

            groundtruth_answers += batch['batch_3m']['answers']
        

        pbar.update(1)

    pbar.close()


    all_predicted_answers_2m = [ans for ans_ls in all_gather_list(predicted_answers_2m)  for ans in ans_ls]
    all_predicted_answers_3m = [ans for ans_ls in all_gather_list(predicted_answers_3m)  for ans in ans_ls]
    all_groundtruth_answers = [ans for ans_ls in all_gather_list(groundtruth_answers)  for ans in ans_ls]
    
    model.train()
    if hvd.rank() != 0:
        return 
    
    total_num = len(all_predicted_answers_2m)
    assert len(all_groundtruth_answers) == total_num
    LOGGER.info('total {} questions has been tested'.format(total_num))
    accurate_num_2m = sum([all_predicted_answers_2m[i] == all_groundtruth_answers[i] for i in range(total_num)])
    accuracy_2m = accurate_num_2m / total_num
    accurate_num_3m = sum([all_predicted_answers_3m[i] == all_groundtruth_answers[i] for i in range(total_num)])
    accuracy_3m = accurate_num_3m / total_num
    tot_time = time()-st
    LOGGER.info(f"evaluation finished in {int(tot_time)} seconds")
    return {'metric_2m':{'accuracy':round(accuracy_2m*100,2)},
            'metric_3m':{'accuracy':round(accuracy_3m*100,2)}}















def str2bool(b):
    if b.lower() in ["false"]:
        return False
    elif b.lower() in ["true"]:
        return True
    elif b is None:
        return None
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
    parser.add_argument("--val_batch_size", default=128, type=int,
                        help="Total batch size for training. "
                             "(batch by examples)")
    parser.add_argument("--negative_size", default=1, type=int,
                        help="Number of negative samples per positive sample")
    parser.add_argument("--inf_minibatch_size", default=400, type=int,
                        help="batch size for running inference. "
                             "(used for validation, and evaluation)")

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
    parser.add_argument('--beam_size', type=int, default=3,
                        help="random seed for initialization")




    parser.add_argument('--sample_frame', type=int, default=0,
                        help="random seed for initialization")
    

    
    parser.add_argument('--resolution', type=int, default=0,
                        help="random seed for initialization")

    parser.add_argument("--audio_path", default='', type=str,
                        help="The output directory where the model "
                             "checkpoints will be written.")
    parser.add_argument('--patch_size', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument('--drop_ratio', type=float, default=0.,
                        help="random seed for initialization")

    parser.add_argument('--video_encoder_type', type=str, default=None,
                        help="random seed for initialization")
    parser.add_argument('--videoswin_timestride', type=int, default=1,
                        help="random seed for initialization")
    parser.add_argument('--multimodal_layernum', type=int, default=2,
                        help="random seed for initialization")

    parser.add_argument('--reuse_embedding', type=str2bool, default=None,
                        help="random seed for initialization")
    parser.add_argument('--average_video', type=str2bool, default=None,
                        help="random seed for initialization")
    parser.add_argument('--pretrain_dir', type=str, default=None,
                        help="random seed for initialization")
    parser.add_argument('--multimodal_norm_mode', type=str, default=None,
                        help="random seed for initialization")

    parser.add_argument('--use_audio', type=str2bool, default=True,
                        help="random seed for initialization")

    parser.add_argument('--strengthen_two', type=str2bool, default=None,
                        help="random seed for initialization")     
    parser.add_argument('--woaudioweight', type=str2bool, default=False,
                        help="random seed for initialization")    
    parser.add_argument('--audio_melbins', type=int, default=None,
                        help="random seed for initialization")

    parser.add_argument('--audio_target_length', type=int, default=None,
                        help="random seed for initialization")
                        
    parser.add_argument('--audio_patch_size', type=int, default=None,
                        help="random seed for initialization")

    parser.add_argument('--audio_frame_shift', type=int, default=None,
                        help="random seed for initialization")  

    parser.add_argument('--average_video_mode', type=str, default=None,
                        help="random seed for initialization")

    parser.add_argument('--average_audio_mode', type=str, default=None,
                        help="random seed for initialization")      
    parser.add_argument('--audio_encoder_weights', type=str, default=None,
                        help="random seed for initialization")                
    args = parse_with_config(parser)


    # if exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not "
    #                      "empty.".format(args.output_dir))

    # options safe guard


    main(args)
