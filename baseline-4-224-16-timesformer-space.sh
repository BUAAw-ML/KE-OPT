basedir="./output/v2.0_final_baseline_4_224_16_timesformer_space"
checkpoint='./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero.pt'


sample_num=4
resolution=224
patch_size=16

# ## finetune on msrvtt fast retrieval
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt-b32-2gpu-lr5e-5-2witer.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-12-12-3-prenorm-reuse'  \
# --fp16  \
# --video_encoder_type 'timesformer_space' \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --checkpoint $checkpoint  &



# ## finetune on msrvtt caption
# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python ./train_caption.py \
# --config ./config/caption-msrvtt-b32-2gpu-lr5e-5-2witer-mlm0.6.json \
# --output_dir $basedir'/caption-msrvtt-12-12-3-prenorm-reuse'   \
# --fp16 \
# --video_encoder_type 'timesformer_space' \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --checkpoint $checkpoint  &


# ### finetune on msrvtt open-ended vqa
# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt-b32-2gpu-lr5e-5-2witer-answerfull.json \
# --output_dir $basedir'/openended-vqa-msrvtt-12-12-3-prenorm-reuse'  \
# --fp16 \
# --video_encoder_type 'timesformer_space' \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --checkpoint $checkpoint 



## finetune on msrvtt caption
CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python ./train_caption.py \
--config ./config/caption-msrvtt-b32-2gpu-lr5e-5-2witer-mlm0.6.json \
--output_dir $basedir'/sss'   \
--fp16 \
--video_encoder_type 'timesformer_space' \
--sample_frame $sample_num \
--resolution $resolution \
--patch_size $patch_size \
--checkpoint "/home/zhongguokexueyuanzidonghuayanjiusuo/shchen/videoOPT-Three/output/v2.0-final-pretrain-webvid-2modal-b96-4gpu-5.5w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm/caption-msrvtt-1-fps1_4-224-16/ckpt/model_step_20000.pt"  \
--zero_shot


### finetune on msrvtt fast retrieval


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 horovodrun -np 6 python ./train_slow_retrieval.py \
# --config ./config/slow-retrieval-msrvtt-b32-2gpu-lr5e-5-3witer.json \
# --output_dir $basedir'/slow-retrieval-msrvtt-pair'  \
# --fp16  \
# --video_encoder_type 'timesformer_space' \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --checkpoint "/raid/shchen/videoOPT-Three/output/v2.0-last-pretrain-webvid-2modal-b96-4gpu-5.5w-1-224-16-mlm+contra+unimlm+match--fp16-lr1e-4-timesformer-space-sharecaptionencoder/slow-retrieval-msrvtt-1fps1_4-224-16/ckpt/model_step_28000.pt" \
# --zero_shot


# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./train_slow_retrieval.py \
# --config ./config/slow-retrieval-msrvtt-b16-4gpu-lr5e-5-3witer.json \
# --output_dir $basedir'/slow-retrieval-msrvtt-pair'  \
# --fp16  \
# --video_encoder_type 'timesformer_space' \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --checkpoint $checkpoint \
# --match_mode 'pair' 

# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./train_slow_retrieval.py \
# --config ./config/slow-retrieval-msrvtt-b16-4gpu-lr5e-5-3witer.json \
# --output_dir $basedir'/slow-retrieval-msrvtt-pair-hard'  \
# --fp16  \
# --video_encoder_type 'timesformer_space' \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --checkpoint $checkpoint \
# --match_mode 'pair_hard' 

# CUDA_VISIBLE_DEVICES=2,3,4,5 horovodrun -np 4 python ./train_slow_retrieval.py \
# --config ./config/slow-retrieval-msrvtt-b16-4gpu-lr5e-5-3witer.json \
# --output_dir $basedir'/slow-retrieval-msrvtt-group'  \
# --fp16  \
# --video_encoder_type 'timesformer_space' \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --checkpoint $checkpoint \
# --match_mode 'group' 


# CUDA_VISIBLE_DEVICES=2,3,4,5 horovodrun -np 4 python ./train_slow_retrieval.py \
# --config ./config/slow-retrieval-msrvtt-b16-4gpu-lr5e-5-3witer.json \
# --output_dir $basedir'/slow-retrieval-msrvtt-group-hard'  \
# --fp16  \
# --video_encoder_type 'timesformer_space' \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --checkpoint $checkpoint \
# --match_mode 'group_hard' 

# ### finetune on msrvtt fast MCQA
# CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python ./train_fast_retrieval.py \
# --config ./config/MCQA-fast-msrvtt-b32-2gpu-lr5e-5-1witer.json \
# --output_dir $basedir'/MCQA-fast-msrvtt'  \
# --fp16 \
# --video_encoder_type 'timesformer_space' \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --checkpoint $checkpoint &




# ## finetune on msrvtt caption
# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python ./train_caption.py \
# --config ./config/caption-msrvtt-b32-2gpu-lr5e-5-2witer-mlm0.6.json \
# --output_dir $basedir'/caption-msrvtt'   \
# --fp16 \
# --video_encoder_type 'timesformer_space' \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --checkpoint "/home/zhongguokexueyuanzidonghuayanjiusuo/shchen/videoOPT-Three/output/v2.0_last_baseline_4_224_16_timesformer_space/caption-msrvtt/ckpt/model_step_20000.pt"  \
# --zero_shot






# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./train_slow_retrieval.py \
# --config ./config/slow-retrieval-msrvtt-4gpu-b8-lr5e-5-3witer.json \
# --output_dir $basedir'/slow-retrieval-msrvtt'  \
# --fp16 \
# --video_encoder_type 'timesformer' \
# --checkpoint $checkpoint &











# ### finetune on msrvtt slow MCQA
# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python ./train_slow_retrieval.py \
# --config ./config/MCQA-slow-msrvtt-b64-lr1e-5-6witer-4-224-32.json \
# --output_dir $basedir'/MCQA-slow-msrvtt' \
# --fp16 \
# --checkpoint "./output/pretrianed_weights/ima21k_vit_base_224_32.pt" &