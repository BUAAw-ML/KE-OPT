"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

optimizer learning rate scheduling helpers
"""
from math import ceil
import ipdb

def noam_schedule(step, warmup_step=4000):
    """ original Transformer schedule"""
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)


def warmup_linear(step, warmup_step, tot_step):
    """ BERT schedule """
    if step < warmup_step:
        return step / warmup_step
    return max(0, (tot_step-step)/(tot_step-warmup_step))


def vqa_schedule(step, warmup_interval, decay_interval,
                 decay_start, decay_rate):
    """ VQA schedule from MCAN """
    if step < warmup_interval:
        return 1/4
    elif step < 2 * warmup_interval:
        return 2/4
    elif step < 3 * warmup_interval:
        return 3/4
    elif step >= decay_start:
        num_decay = ceil((step - decay_start) / decay_interval)
        return decay_rate ** num_decay
    else:
        return 1


def get_lr_sched(global_step, opts):
    # learning rate scheduling
    scheduler = opts.lr_scheduler if hasattr(opts,'lr_scheduler')  else 'linear'
    assert opts.warmup_ratio < 1
    warmup_steps = opts.warmup_ratio * opts.num_train_steps
    #warmup_steps = opts.num_train_steps * 0.05
    learning_rate = opts.learning_rate

    if scheduler == 'linear':
        lr_this_step = learning_rate * warmup_linear(
            global_step, warmup_steps, opts.num_train_steps)

    elif scheduler == 'linear_keep':
        keep_start_step = int(opts.keep_start_ratio * opts.num_train_steps)
        keep_value = opts.keep_value  
        if global_step < keep_start_step:
            lr_this_step = max( learning_rate * warmup_linear(
            global_step, warmup_steps, keep_start_step), keep_value)
        else:
            lr_this_step = keep_value
        
    elif scheduler =='noam':
        factor = learning_rate /(warmup_steps) ** (-0.5)
        lr_this_step = factor * min((global_step ) ** (-0.5), (global_step ) * warmup_steps ** (-1.5))

    elif scheduler =='multi_step':
        steps_num = opts.steps_num
        step_size = opts.num_train_steps // steps_num
        steps = list(range(0,opts.num_train_steps,step_size))[1:]

        factor = opts.step_factor
        times = 0
        if global_step < warmup_steps:
            lr_this_step = global_step/warmup_steps * learning_rate
        else:
            for s in steps:
                if global_step >= s:
                    times += 1
            lr_this_step = learning_rate * (factor ** times)
    
    
    elif scheduler == 'fix':
        lr_this_step = learning_rate
    
    else:
        raise NotImplementedError
                     
    if lr_this_step <= 0:
        lr_this_step = 1e-8
    return lr_this_step
