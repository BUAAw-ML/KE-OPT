# basedir="./output/pretrain-webvid-2modal-b250-4gpu-20w-mlm-match-contra-unimlm"
# checkpoint_step=200000
basedir='./output/v3.0-baseline-4-224-16-timesformer-space'
fps_or_frames=4
sample_num=4
resolution=224
patch_size=16
learning_rate=5e-5
ngpu=2
train_batch_size=32
num_train_steps=20000
weight_decay=0.001
video_encoder_type='timesformer_space'
checkpoint="./output/pretrianed_weights/v3.0_bertbaseuncased_timesformer16_timezero_ast32.pt"
use_audio='false'

if [ $fps_or_frames -ge 5 ]
then 
    echo $fps_or_frames
    prefix='frames'
    video_path='./datasets/msrvtt/frames_'$fps_or_frames
else 
    prefix='fps'
    video_path='./datasets/msrvtt/frames_fps'$fps_or_frames
fi

# finetune on msrvtt fast retrieval
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np $ngpu python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio   \
# --checkpoint $checkpoint \
# --video_path $video_path  \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size $train_batch_size \
# --learning_rate $learning_rate \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio $use_audio \
# --fp16   &

finetune on msrvtt fast retrieval
CUDA_VISIBLE_DEVICES=0,1 horovodrun -np $ngpu python ./train_fast_retrieval.py \
--config ./config/fast-retrieval-msrvtt.json \
--output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'--seed70'   \
--checkpoint $checkpoint \
--video_path $video_path  \
--sample_frame $sample_num \
--resolution $resolution \
--patch_size $patch_size \
--train_batch_size $train_batch_size \
--learning_rate $learning_rate \
--num_train_steps $num_train_steps \
--weight_decay $weight_decay \
--video_encoder_type $video_encoder_type \
--use_audio $use_audio \
--seed 70 \
--fp16   &
finetune on msrvtt fast retrieval
CUDA_VISIBLE_DEVICES=2,3 horovodrun -np $ngpu python ./train_fast_retrieval.py \
--config ./config/fast-retrieval-msrvtt.json \
--output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'--seed80'   \
--checkpoint $checkpoint \
--video_path $video_path  \
--sample_frame $sample_num \
--resolution $resolution \
--patch_size $patch_size \
--train_batch_size $train_batch_size \
--learning_rate $learning_rate \
--num_train_steps $num_train_steps \
--weight_decay $weight_decay \
--video_encoder_type $video_encoder_type \
--use_audio $use_audio \
--seed 80 \
--fp16   

# ## finetune on msrvtt caption
# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np $ngpu python ./train_caption.py \
# --config ./config/caption-msrvtt.json \
# --output_dir $basedir'/caption-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio   \
# --checkpoint $checkpoint \
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
# --fp16  

# ## finetune on msrvtt open-ended vqa
# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np $ngpu python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt.json \
# --output_dir $basedir'/openended-vqa-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio   \
# --checkpoint $checkpoint \
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
# --fp16  

# ## finetune on vatex zh caption
# CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python ./train_caption.py \
# --config ./config/caption-vatex-zh.json \
# --output_dir $basedir'/caption-vatexzh-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio   \
# --checkpoint './output/pretrianed_weights/double_modify_timesformerbase16_bertbasechinese_timezero.pt' \
# --video_path './datasets/vatex/frames_fps4'  \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size $train_batch_size \
# --learning_rate $learning_rate \
# --num_train_steps 40000 \
# --weight_decay $weight_decay \
# --use_audio $use_audio \
# --video_encoder_type $video_encoder_type \
# --fp16  

# ### finetune on msrvtt slow retrieval
# CUDA_VISIBLE_DEVICES=6,7 horovodrun -np $ngpu python ./train_slow_retrieval.py \
# --config ./config/slow-retrieval-msrvtt.json \
# --output_dir $basedir'/slow-retrieval-msrvtt'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay   \
# --checkpoint $checkpoint \
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


