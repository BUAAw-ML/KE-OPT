# basedir="./output/pretrain-webvid-2modal-b250-4gpu-20w-mlm-match-contra-unimlm"
# checkpoint_step=200000
basedir=$1
checkpoint_step=$2
fps_or_frames=1
sample_num=4
resolution=224
patch_size=32
checkpoint_path=$basedir'/ckpt/model_step_'$checkpoint_step'.pt'  

if [ $fps_or_frames -ge 5 ]
then 
    echo $fps_or_frames
    prefix='frames'
    video_path='./datasets/msrvtt/frames_'$fps_or_frames
else 
    prefix='fps'
    video_path='./datasets/msrvtt/frames_fps'$fps_or_frames
fi

### finetune on msrvtt fast retrieval
CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python ./train_fast_retrieval.py \
--config ./config/fast-retrieval-msrvtt-b32-2gpu-lr5e-5-2witer.json \
--output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size   \
--checkpoint $checkpoint_path \
--video_path $video_path  \
--sample_frame $sample_num \
--resolution $resolution \
--patch_size $patch_size \
--video_encoder_type 'videoswin' \
--videoswin_timestride 1 \
--fp16 &


## finetune on msrvtt caption
CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python ./train_caption.py \
--config ./config/caption-msrvtt-b32-2gpu-lr5e-5-2witer-mlm0.6.json \
--output_dir $basedir'/caption-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size   \
--checkpoint $checkpoint_path \
--video_path $video_path   \
--sample_frame $sample_num \
--resolution $resolution \
--patch_size $patch_size \
--video_encoder_type 'videoswin' \
--videoswin_timestride 1 \
--fp16  &

## finetune on msrvtt open-ended vqa
CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python ./train_open_ended_vqa.py \
--config ./config/VQA-msrvtt-b32-2gpu-lr5e-5-2witer-answerfull.json \
--output_dir $basedir'/openended-vqa-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size  \
--checkpoint $checkpoint_path \
--video_path $video_path   \
--sample_frame $sample_num \
--resolution $resolution \
--patch_size $patch_size \
--video_encoder_type 'videoswin' \
--videoswin_timestride 1 \
--fp16  &

### finetune on msrvtt fast MCQA
CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python ./train_fast_retrieval.py \
--config ./config/MCQA-fast-msrvtt-b32-2gpu-lr5e-5-1witer.json \
--output_dir $basedir'/MCQA-fast-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size    \
--checkpoint $checkpoint_path \
--video_path $video_path   \
--sample_frame $sample_num \
--resolution $resolution \
--patch_size $patch_size \
--video_encoder_type 'videoswin' \
--videoswin_timestride 1 \
--fp16  &


# ### finetune on msrvtt slow retrieval
# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./train_slow_retrieval.py \
# --config ./config/slow-retrieval-msrvtt-4gpu-b8-lr5e-5-3witer.json \
# --output_dir $basedir'/slow-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size    \
# --checkpoint $checkpoint_path \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --video_encoder_type 'videoswin' \
# --fp16  &


# finetune on msrvtt slow MCQA
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python ./train_slow_retrieval.py \
# --config ./config/MCQA-slow-msrvtt-b64-lr1e-5-6witer.json \
# --output_dir $basedir'/MCQA-slow-msrvtt'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size  \
# --checkpoint $checkpoint_path \
# --fp16  &



#sh auto_finetune-4-224-32.sh "./output/v2.0-clstoken-pretrain-webvid-2modal-b150-4gpu-8w-4-224-32-unimlm-contra-fp16-lr1e-4-vitpretrained/" 80000 30 2

