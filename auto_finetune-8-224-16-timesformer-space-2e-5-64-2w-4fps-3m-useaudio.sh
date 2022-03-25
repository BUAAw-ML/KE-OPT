# basedir="./output/pretrain-webvid-2modal-b250-4gpu-20w-mlm-match-contra-unimlm"
# checkpoint_step=200000
basedir=$1
fps_or_frames=4
sample_num=8
resolution=224
patch_size=16
learning_rate=2e-5
ngpu=8
train_batch_size=8
num_train_steps=20000
weight_decay=0.001
use_audio='true'
video_encoder_type='timesformer_space'


if [ $fps_or_frames -ge 5 ]
then 
    echo $fps_or_frames
    prefix='frames'
    video_path='./datasets/msrvtt/frames_'$fps_or_frames
else 
    prefix='fps'
    video_path='./datasets/msrvtt/frames_fps'$fps_or_frames
fi

## finetune on msrvtt fast retrieval



CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python ./train_fast_retrieval.py \
--config ./config/fast-retrieval-msrvtt.json \
--output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr-2e-5-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio   \
--video_path $video_path  \
--sample_frame $sample_num \
--resolution $resolution \
--patch_size $patch_size \
--train_batch_size $train_batch_size \
--learning_rate 2e-5 \
--num_train_steps $num_train_steps \
--weight_decay $weight_decay \
--video_encoder_type $video_encoder_type \
--use_audio $use_audio \
--test_ids_path './datasets/msrvtt/1kAsplit_test_id.json' \
--zero_shot \
--checkpoint "/raid/shchen/videoOPT-Three/output/v3.0_start_pretrain-webvid+cc3m+audioset-3modal-7w-1:1:1-1-224-16-spaceaverage-strengthentwo/fast-retrieval-msrvtt-fps4_8-224-16-lr-2e-5-bs8-gpu8-step20000-wd0.001-useaudiotrue/ckpt/model_step_20000.pt" \
--fp16  


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np $ngpu python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr-2e-5-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio   \
# --video_path $video_path  \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size $train_batch_size \
# --learning_rate 2e-5 \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio $use_audio \
# --test_ids_path './datasets/msrvtt/1kAsplit_test_id.json' \
# --pretrain_dir $basedir \
# --fp16  



# ## finetune on msrvtt caption
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np $ngpu python ./train_caption.py \
# --config ./config/caption-msrvtt.json \
# --output_dir $basedir'/caption-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay   \
# --pretrain_dir $basedir \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size $train_batch_size \
# --learning_rate $learning_rate \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio $use_audio \
# --strengthen_two true \
# --fp16   

## finetune on msrvtt open-ended vqa
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np $ngpu python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt.json \
# --output_dir $basedir'/openended-vqa-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay   \
# --pretrain_dir $basedir \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size $train_batch_size \
# --learning_rate $learning_rate \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio $use_audio \
# --strengthen_two true \
# --fp16  



# ### finetune on msrvtt slow retrieval
# CUDA_VISIBLE_DEVICES=6,7 horovodrun -np $ngpu python ./train_slow_retrieval.py \
# --config ./config/slow-retrieval-msrvtt.json \
# --output_dir $basedir'/slow-retrieval-msrvtt'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay   \
# --pretrain_dir $basedir \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size $train_batch_size \
# --learning_rate $learning_rate \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --fp16  


# ## finetune on msrvtt caption
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python ./train_caption.py \
# --config ./config/caption-msrvtt-b32-2gpu-lr5e-5-2witer-mlm0.6.json \
# --output_dir $basedir'/caption-msrvtt-1-averagevideo'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size   \
# --checkpoint $checkpoint_path \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --video_encoder_type 'timesformer_space' \
# --average_video True \
# --fp16  &

# ## finetune on msrvtt open-ended vqa
# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt-b32-2gpu-lr5e-5-2witer-answerfull.json \
# --output_dir $basedir'/openended-vqa-msrvtt-1-averagevideo'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size  \
# --checkpoint $checkpoint_path \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --average_video True \
# --video_encoder_type 'timesformer_space' \
# --fp16  &

# ### finetune on msrvtt slow retrieval grouphard
# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./train_slow_retrieval.py \
# --config ./config/slow-retrieval-msrvtt-b16-4gpu-lr5e-5-3witer.json \
# --output_dir $basedir'/slow-retrieval-msrvtt-1-grouphard-averagevideo'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size    \
# --fp16  \
# --checkpoint $checkpoint_path \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --match_mode 'group_hard' \
# --video_encoder_type 'timesformer_space' \
# --average_video True \
# --fp16  

# ### finetune on msrvtt slow retrieval grouphard
# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./train_slow_retrieval.py \
# --config ./config/slow-retrieval-msrvtt-b16-4gpu-lr5e-5-3witer.json \
# --output_dir $basedir'/slow-retrieval-msrvtt-1-grouphard'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size    \
# --fp16  \
# --checkpoint $checkpoint_path \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --match_mode 'group_hard' \
# --video_encoder_type 'timesformer_space' \
# --fp16  






# ### finetune on msrvtt fast MCQA
# CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python ./train_fast_retrieval.py \
# --config ./config/MCQA-fast-msrvtt-b32-2gpu-lr5e-5-1witer.json \
# --output_dir $basedir'/MCQA-fast-msrvtt-1-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size    \
# --checkpoint $checkpoint_path \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --video_encoder_type 'timesformer_space' \
# --fp16  &


# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt-b32-2gpu-lr5e-5-2witer.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-contrasync-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size   \
# --checkpoint $checkpoint_path \
# --video_path $video_path  \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --video_encoder_type 'timesformer_space' \
# --contra_sync true \
# --fp16 &


# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt-b32-2gpu-lr5e-5-2witer.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-dualsoftmax-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size   \
# --checkpoint $checkpoint_path \
# --video_path $video_path  \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --video_encoder_type 'timesformer_space' \
# --use_dualsoftmax true \
# --fp16 &

# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt-b32-2gpu-lr5e-5-2witer.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-contrasync-dualsoftmax-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size   \
# --checkpoint $checkpoint_path \
# --video_path $video_path  \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --video_encoder_type 'timesformer_space' \
# --use_dualsoftmax true \
# --contra_sync true \
# --fp16 &


# ### finetune on msrvtt slow retrieval
# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./train_slow_retrieval.py \
# --config ./config/slow-retrieval-msrvtt-4gpu-b8-lr5e-5-3witer.json \
# --output_dir $basedir'/slow-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size    \
# --checkpoint $checkpoint_path \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --video_encoder_type 'timesformer' \
# --fp16  


# finetune on msrvtt slow MCQA
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python ./train_slow_retrieval.py \
# --config ./config/MCQA-slow-msrvtt-b64-lr1e-5-6witer.json \
# --output_dir $basedir'/MCQA-slow-msrvtt'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size  \
# --checkpoint $checkpoint_path \
# --fp16  &



#sh auto_finetune-4-224-32.sh "./output/v2.0-clstoken-pretrain-webvid-2modal-b150-4gpu-8w-4-224-32-unimlm-contra-fp16-lr1e-4-vitpretrained/" 80000 30 2


