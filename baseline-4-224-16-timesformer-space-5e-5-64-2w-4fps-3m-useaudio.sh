# basedir="./output/pretrain-webvid-2modal-b250-4gpu-20w-mlm-match-contra-unimlm"
# checkpoint_step=200000
basedir='./output/v3.0-baseline-4-224-16-timesformer-space'
fps_or_frames=4
sample_num=4
resolution=224
patch_size=16
learning_rate=5e-5
ngpu=4
train_batch_size=16
num_train_steps=20000
weight_decay=0.001
video_encoder_type='timesformer_space'
#checkpoint="./output/pretrianed_weights/v3.0_bertbaseuncased_timesformer16_timezero_ast32.pt"
checkpoint=None
use_audio='true'

if [ $fps_or_frames -ge 5 ]
then 
    echo $fps_or_frames
    prefix='frames'
    video_path='./datasets/msrvtt/frames_'$fps_or_frames
else 
    prefix='fps'
    video_path='./datasets/msrvtt/frames_fps'$fps_or_frames
fi

# #finetune on msrvtt fast retrieval
# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np $ngpu python ./train_fast_retrieval.py \
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

#finetune on msrvtt fast retrieval
# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np $ngpu python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'smalllr-testaudo'   \
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
# --test_ids_path './datasets/msrvtt/1kAsplit_test_id_withaudio.json' \
# --fp16 

# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np $ngpu python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'-testaudo_v+a_independtemp'   \
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
# --test_ids_path './datasets/msrvtt/1kAsplit_test_id_withaudio.json' \
# --fp16   

# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'-multitrans_wovata'   \
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
# --use_multimodal_encoder True \
# --test_ids_path './datasets/msrvtt/1kAsplit_test_id_withaudio.json' \
# --fp16   

# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np $ngpu python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'-multitrans_withvata'   \
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
# --use_multimodal_encoder True \
# --with_vata_loss True \
# --test_ids_path './datasets/msrvtt/1kAsplit_test_id_withaudio.json' \
# --fp16   &


# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np $ngpu python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'-concate-fulltest'   \
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
# --va_fusion_mode 'concate' \
# --test_ids_path './datasets/msrvtt/1kAsplit_test_id.json' \
# --fp16   &

# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np $ngpu python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'-concate-fulltest-va_all'   \
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
# --va_fusion_mode 'concate' \
# --va_all true \
# --test_ids_path './datasets/msrvtt/1kAsplit_test_id.json' \
# --fp16  

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'8-'$resolution'-'$patch_size'-lr-2e-5-bs8-gpu-8-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'-concate-fulltest-va_all-2mft'   \
# --video_path $video_path  \
# --sample_frame 8 \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size $train_batch_size \
# --learning_rate 2e-5 \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio $use_audio \
# --va_fusion_mode 'concate' \
# --va_all true \
# --test_ids_path './datasets/msrvtt/1kAsplit_test_id.json' \
# --checkpoint /raid/shchen/videoOPT-Three/output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch/ckpt/model_step_60000.pt \
# --fp16  

CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np $ngpu python ./train_fast_retrieval.py \
--config ./config/fast-retrieval-msrvtt.json \
--output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr-2e-5-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'-concate-fulltest-va_all-2mft-withtawaloss-0.1'   \
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
--va_fusion_mode 'concate' \
--va_all true \
--test_ids_path './datasets/msrvtt/1kAsplit_test_id.json' \
--checkpoint /raid/shchen/videoOPT-Three/output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch/ckpt/model_step_60000.pt \
--with_vata_loss True \
--fp16  &










# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np $ngpu python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'-testaudo_v+a_wovata'   \
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
# --test_ids_path './datasets/msrvtt/1kAsplit_test_id_withaudio.json' \
# --fp16   

# #finetune on msrvtt fast retrieval
# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np $ngpu python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'--va_all'   \
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
# --va_all true \
# --fp16   


# #finetune on msrvtt fast retrieval
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr2e-5-bs32-gpu2-step'$num_train_steps'-wd'$weight_decay'-useaudio-false-twoft'   \
# --video_path $video_path  \
# --sample_frame $sample_num \
# --checkpoint /raid/shchen/videoOPT-Three/output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch/ckpt/model_step_60000.pt \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size 32 \
# --learning_rate 2e-5 \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio false \
# --fp16   &

# ####this is start
# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python ./train_caption.py \
# --config ./config/caption-msrvtt.json \
# --output_dir $basedir'/caption-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr2e-5-bs32-gpu2-step'$num_train_steps'-wd'$weight_decay'-useaudio-false-twoft'   \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --checkpoint /raid/shchen/videoOPT-Three/output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch/ckpt/model_step_60000.pt \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size 32 \
# --learning_rate 2e-5 \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio false \
# --fp16  &

# # finetune on msrvtt open-ended vqa
# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt.json \
# --output_dir $basedir'/openended-vqa-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr2e-5-bs32-gpu2-step'$num_train_steps'-wd'$weight_decay'-useaudio-false-twoft'   \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --checkpoint /raid/shchen/videoOPT-Three/output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch/ckpt/model_step_60000.pt \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size 32 \
# --learning_rate 2e-5 \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio false \
# --fp16  &


# CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr2e-5-bs32-gpu2-step'$num_train_steps'-wd'$weight_decay'-useaudio-false-twoft-seed70'   \
# --video_path $video_path  \
# --sample_frame $sample_num \
# --checkpoint /raid/shchen/videoOPT-Three/output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch/ckpt/model_step_60000.pt \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size 32 \
# --seed 70 \
# --learning_rate 2e-5 \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio false \
# --fp16   

# ####this is start
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python ./train_caption.py \
# --config ./config/caption-msrvtt.json \
# --output_dir $basedir'/caption-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr2e-5-bs32-gpu2-step'$num_train_steps'-wd'$weight_decay'-useaudio-false-twoft-seed70'   \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --checkpoint /raid/shchen/videoOPT-Three/output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch/ckpt/model_step_60000.pt \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size 32 \
# --learning_rate 2e-5 \
# --seed 70 \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio false \
# --fp16  &

# # finetune on msrvtt open-ended vqa
# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt.json \
# --output_dir $basedir'/openended-vqa-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr2e-5-bs32-gpu2-step'$num_train_steps'-wd'$weight_decay'-useaudio-false-twoft-seed70'   \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --checkpoint /raid/shchen/videoOPT-Three/output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch/ckpt/model_step_60000.pt \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size 32 \
# --learning_rate 2e-5 \
# --seed 70 \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio false \
# --fp16  &


# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr2e-5-bs32-gpu2-step'$num_train_steps'-wd'$weight_decay'-useaudio-false-twoft-seed80'   \
# --video_path $video_path  \
# --sample_frame $sample_num \
# --checkpoint /raid/shchen/videoOPT-Three/output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch/ckpt/model_step_60000.pt \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size 32 \
# --seed 80 \
# --learning_rate 2e-5 \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio false \
# --fp16   

# ####this is start
# CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python ./train_caption.py \
# --config ./config/caption-msrvtt.json \
# --output_dir $basedir'/caption-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr2e-5-bs32-gpu2-step'$num_train_steps'-wd'$weight_decay'-useaudio-false-twoft-seed80'   \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --checkpoint /raid/shchen/videoOPT-Three/output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch/ckpt/model_step_60000.pt \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size 32 \
# --learning_rate 2e-5 \
# --seed 80 \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio false \
# --fp16  
# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np $ngpu python ./train_caption.py \
# --config ./config/caption-msrvtt.json \
# --output_dir $basedir'/caption-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr2e-5-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio-false-twoft'   \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --checkpoint /raid/shchen/videoOPT-Three/output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch/ckpt/model_step_60000.pt \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size $train_batch_size \
# --learning_rate 2e-5 \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio false \
# --fp16  &

# # finetune on msrvtt open-ended vqa
# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np $ngpu python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt.json \
# --output_dir $basedir'/openended-vqa-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr2e-5-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio-false-twoft'   \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --checkpoint /raid/shchen/videoOPT-Three/output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch/ckpt/model_step_60000.pt \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size $train_batch_size \
# --learning_rate 2e-5 \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio false \
# --fp16  

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python ./train_caption.py \
# --config ./config/caption-msrvtt.json \
# --output_dir $basedir'/caption-msrvtt-'$prefix$fps_or_frames'_8-'$resolution'-'$patch_size'-lr2e-5-bs8-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'twoft'   \
# --video_path $video_path   \
# --sample_frame $sample_num \
# --checkpoint /raid/shchen/videoOPT-Three/output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch/ckpt/model_step_60000.pt \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size 8 \
# --learning_rate 2e-5 \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio $use_audio \
# --fp16  



# # finetune on msrvtt caption
# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np $ngpu python ./train_caption.py \
# --config ./config/caption-msrvtt.json \
# --output_dir $basedir'/caption-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio   \
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
# --fp16  &
############ audio representation
# finetune on msrvtt open-ended vqa
# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np $ngpu python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt.json \
# --output_dir $basedir'/openended-vqa-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'average-video-audio-space'   \
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
# --average_video_mode 'space' \
# --average_audio_mode 'space' \
# --fp16  

# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np $ngpu python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt.json \
# --output_dir $basedir'/openended-vqa-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'average-video-audio-space-audiomelbins64'   \
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
# --average_video_mode 'space' \
# --average_audio_mode 'space' \
# --audio_melbins 64 \
# --fp16  

# finetune on msrvtt caption
# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np $ngpu python ./train_caption.py \
# --config ./config/caption-msrvtt.json \
# --output_dir $basedir'/caption-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'average-video-audio-space'   \
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
# --average_video_mode 'space' \
# --average_audio_mode 'space' \
# --fp16  &

CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np $ngpu python ./train_open_ended_vqa.py \
--config ./config/VQA-msrvtt.json \
--output_dir $basedir'/openended-vqa-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'average-video-audio-space-audiomelbins64-frameshift2-tarlen512-patch32'   \
--video_path $video_path   \
--sample_frame $sample_num \
--resolution $resolution \
--patch_size $patch_size \
--train_batch_size $train_batch_size \
--learning_rate $learning_rate \
--num_train_steps $num_train_steps \
--weight_decay $weight_decay \
--video_encoder_type $video_encoder_type \
--use_audio $use_audio \
--average_video_mode 'space' \
--average_audio_mode 'space' \
--audio_melbins 64 \
--audio_frame_shift 20 \
--audio_target_length 512 \
--audio_patch_size  32 \
--audio_encoder_weights '/raid/shchen/videoOPT-Three/output/pretrianed_weights/ViT-B_32.npz'
--fp16  

# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np $ngpu python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt.json \
# --output_dir $basedir'/openended-vqa-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'average-video-audio-space-target512'   \
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
# --average_video_mode 'space' \
# --average_audio_mode 'space' \
# --audio_target_length 512 \
# --fp16  &


# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np $ngpu python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt.json \
# --output_dir $basedir'/openended-vqa-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'average-video-audio-space-frameshift20-target512'   \
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
# --average_video_mode 'space' \
# --average_audio_mode 'space' \
# --audio_frame_shift 20 \
# --audio_target_length 512 \
# --fp16  




# # finetune on msrvtt caption
# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np $ngpu python ./train_caption.py \
# --config ./config/caption-msrvtt.json \
# --output_dir $basedir'/caption-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'--woaudioweight'   \
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
# --woaudioweight true \
# --fp16  &

# # finetune on msrvtt open-ended vqa
# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np $ngpu python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt.json \
# --output_dir $basedir'/openended-vqa-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'--woaudioweight'   \
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
# --woaudioweight true \
# --fp16  
# # finetune on msrvtt caption
# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np $ngpu python ./train_caption.py \
# --config ./config/caption-msrvtt.json \
# --output_dir $basedir'/caption-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'-strengthentwofalse'   \
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
# --strengthen_two false \
# --fp16   

## finetune on msrvtt open-ended vqa
# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np $ngpu python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt.json \
# --output_dir $basedir'/openended-vqa-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'-strengthentwofalse'   \
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
# --strengthen_two false \
# --fp16  &

# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np $ngpu python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio'--va_all-contradim256'   \
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
# --va_all true \
# --contra_dim 256 \
# --fp16   


# sh ./baseline-4-224-16-timesformer-space-5e-5-64-2w-4fps-3m-notuseaudio.sh
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


