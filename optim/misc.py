"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc lr helper
"""
from torch.optim import Adam, Adamax

from .adamw import AdamW
import ipdb

def build_optimizer(model, opts):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer



def build_optimizer_for_VQA(model, opts):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_param = 'vqa_head'
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay) and not new_param in n],
         'weight_decay': opts.weight_decay,
         'lr': opts.learning_rate},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay) and not new_param in n],
         'weight_decay': 0.0,
         'lr': opts.learning_rate},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay) and new_param in n],
         'weight_decay': opts.weight_decay,
         'lr': opts.learning_rate*5},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay) and new_param in n],
         'weight_decay': 0.0,
         'lr': opts.learning_rate*5}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                          betas=opts.betas)
    return optimizer
