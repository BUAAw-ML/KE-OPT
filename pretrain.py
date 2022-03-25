"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

OPT pre-training
"""
import argparse
import math
import os
from os.path import join
from time import time

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from apex import amp
from horovod import torch as hvd

from tqdm import tqdm
from model.pretrain import OPTForPretraining
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_random_seed

from pretrain_data import create_train_dataloaders, create_val_dataloaders
from pretrain_validate import validate

from data import  MetaLoader, PrefetchLoader , MapperGroup
import ipdb
from data.loader import move_to_cuda 


def main(opts):
    LOGGER.info("start hvd")
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    LOGGER.info("hvd init")
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

    """ resume """
    if opts.resume:
        ckpt_dir = os.path.join(opts.output_dir,'ckpt')
        previous_optimizer_state = [i  for i in os.listdir(ckpt_dir) if i.startswith('optimizer')]
        assert len(previous_optimizer_state)==1
        previous_optimizer_state = previous_optimizer_state[0]
        previous_model_state = previous_optimizer_state.replace('optimizer','model')
        previous_step = int(previous_model_state.split('.')[0].split('_')[-1])
        previous_optimizer_state = os.path.join(ckpt_dir, previous_optimizer_state)
        previous_model_state = os.path.join(ckpt_dir, previous_model_state)
        assert os.path.exists(previous_optimizer_state) and os.path.exists(previous_model_state)
        LOGGER.info("choose previous model: {}".format(previous_model_state))
        LOGGER.info("choose previous optimizer: {}".format(previous_optimizer_state))
    else:
        previous_model_state = ''
        previous_optimizer_state = ''
        previous_step = 0

    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps,initial=previous_step)
        TB_LOGGER.set_step(previous_step)
        model_saver = ModelSaver(join(args.output_dir, 'ckpt'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()


    mapper_group = MapperGroup()
    
    
    train_dataloaders, mapper_group = create_train_dataloaders(opts.data_cfg, opts, mapper_group)

    if opts.use_validate:
        val_dataloaders,_= create_val_dataloaders(opts.data_cfg, opts, mapper_group)

    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=opts.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)




    # Prepare model
    if previous_model_state:
        checkpoint = torch.load(previous_model_state, map_location = device)
        LOGGER.info("Load Checkpoint {}".format(previous_model_state))
    elif opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint, map_location = device)
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
                
        LOGGER.info("Load Checkpoint {}".format(opts.checkpoint))
    else:
        checkpoint = {}
    ### add video_dim to model_cfg

    #opts.model_cfg['video_dim'] = (opts.video_cfg['patch_size'] **2 ) * 3

    model = OPTForPretraining.from_pretrained(
        opts.model_cfg,  checkpoint, opts.video_cfg, opts.audio_cfg)
    #ipdb.set_trace()
    model.to(device)
    model.train()
    #del(checkpoint)
    # make sure every process has same model parameters in the beginning
    LOGGER.info("broadcast_tensors")

    broadcast_tensors([p.data for p in model.parameters()], 0)
    #set_dropout(model, opts.dropout)

    # Prepare optimizer
    LOGGER.info("build_optimizer")
    optimizer = build_optimizer(model, opts)
    if previous_optimizer_state:
        checkpoint_optimizer = torch.load(previous_optimizer_state, map_location = device)
        optimizer.load_state_dict(checkpoint_optimizer)
        #del(checkpoint_optimizer)
    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}
    LOGGER.info("amp.initialize")
    model, optimizer = amp.initialize(model, optimizer,
                                      num_losses=len(task2scaler),
                                      enabled=opts.fp16, opt_level='O2')

    global_step = previous_step
    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)
    LOGGER.info("  Start step = %d", previous_step)

    # to compute training statistics


    loss_moving_averagetors ={}

    grad_norm = 0


    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()


    if opts.use_validate:
        validate(model, val_dataloaders, opts, global_step=previous_step)
    if opts.only_eval:
        return 

    for step, (name, batch) in enumerate(meta_loader):

        
        task = name.split('--')[0]
        loss_dict = model(batch, task=task, compute_loss=True)
        


        loss = sum(list(loss_dict.values()))
        loss_dict['total_loss'] = loss
        loss_dict = {k:v.item() for k,v in loss_dict.items()}
        if opts.gradient_accumulation_steps > 1:
            loss = loss / opts.gradient_accumulation_steps

        delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
        with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale,
                            loss_id=task2scaler[name]) as scaled_loss:
            scaled_loss.backward()
            if not delay_unscale:
                # gather gradients from every processes
                # do this before unscaling to make sure every process uses
                # the same gradient scale
                grads = [p.grad.data for p in model.parameters()
                            if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))

        ####accumulate loss
        if not name in loss_moving_averagetors:
            ### first time initialize 
            for k in loss_dict.keys():
                loss_moving_averagetors[f'loss_{name}/{k}'] = RunningMeter()
        
        for k,v in loss_dict.items():
            loss_moving_averagetors[f'loss_{name}/{k}'](v)



        if (step + 1) % opts.gradient_accumulation_steps == 0:

            global_step += 1
            # learning rate scheduling
            lr_this_step = get_lr_sched(global_step, opts)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step

            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # log loss
            # NOTE: not gathered across GPUs for efficiency
            TB_LOGGER.log_scaler_dict({name: averagetor.val
                                        for name, averagetor in loss_moving_averagetors.items()
                                        if averagetor.val is not None})

            if global_step % 200 == 0:    
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
                LOGGER.info(f'Step {global_step}: start validation')

                if opts.use_validate:
                    validate(model, val_dataloaders, opts, global_step = global_step)

                model_saver.save(model, global_step, optimizer)

            TB_LOGGER.step()


        if global_step >= opts.num_train_steps:
            break
        
    if opts.use_validate:
        validate(model, val_dataloaders, opts, global_step = global_step)
        model_saver.save(model, global_step, optimizer)



def str2bool(b):
    if b.lower() in ["false"]:
        return False
    elif b.lower() in ["true"]:
        return True
    else:
        raise Exception("Invalid Bool Value")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    # NOTE: train tasks and val tasks cannot take command line arguments

    parser.add_argument("--model_cfg", type=str,
                        help="path to model structure config json")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="path to model checkpoint (*.pt)")
    # parser.add_argument("--optimizer_checkpoint", default=None, type=str,
    #                     help="path to optimizer checkpoint (*.pt)")
    #parser.add_argument('--start_step', default=0, type=int, help='use txt out')
    parser.add_argument('--resume', action = 'store_true', help='use txt out')
    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    parser.add_argument('--rand_all_prob', default=0.3, type=float,
                        help='probability to mask in video training')
    parser.add_argument('--mvm_prob', default=0.15, type=float,
                        help='probability to mask in video training')
    parser.add_argument('--mam_prob', default=0.15, type=float,
                        help='probability to mask in video training')
    parser.add_argument('--match_neg_prob', default=0.5, type=float,
                        help='probability to make negative examples'
                             'in VTM training')

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    
    parser.add_argument('--max_video_len', type=int, default=30,
                        help='max number of tokens in text (BERT BPE)')

    parser.add_argument('--max_audio_len', type=int, default=30,
                        help='max number of tokens in text (BERT BPE)')


    # training parameters
    parser.add_argument("--train_batch_size", default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=400000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adamw',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2.0, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', default=True, type=str2bool,
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    #parser.add_argument('--pin_mem', action='store_true', help="pin memory")
    parser.add_argument('--pin_mem', default=False, type=str2bool, help="pin memory")
    # can use config files
    parser.add_argument('--config', required=True, help='JSON config files')


    parser.add_argument('--only_eval', action = 'store_true', help='use txt out')
    parser.add_argument('--use_validate', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--show_time', default=False, type=str2bool, help='use txt out')

    
    parser.add_argument('--beam_size', type=int, default=1,
                        help="random seed for initialization")

    args = parse_with_config(parser)

    # if exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not "
    #                      "empty.".format(args.output_dir))


    main(args)
