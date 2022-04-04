basedir="./output/v2.0_double_modify_new_baseline_4_224_32_videoswin-timestride1_1"
checkpoint='./output/pretrianed_weights/double_modify_videoswinbase32_bertbaseuncased.pt'

## finetune on msrvtt fast retrieval
CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python ./train_fast_retrieval.py \
--config ./config/fast-retrieval-msrvtt-b32-2gpu-lr5e-5-2witer.json \
--output_dir $basedir'/fast-retrieval-msrvtt'  \
--fp16  \
--video_encoder_type 'videoswin' \
--videoswin_timestride 1 \
--checkpoint $checkpoint  &



### finetune on msrvtt caption
CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python ./train_caption.py \
--config ./config/caption-msrvtt-b32-2gpu-lr5e-5-2witer-mlm0.6.json \
--output_dir $basedir'/caption-msrvtt'   \
--fp16 \
--video_encoder_type 'videoswin' \
--videoswin_timestride 1 \
--checkpoint $checkpoint &


### finetune on msrvtt open-ended vqa
CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python ./train_open_ended_vqa.py \
--config ./config/VQA-msrvtt-b32-2gpu-lr5e-5-2witer-answerfull.json \
--output_dir $basedir'/openended-vqa-msrvtt'  \
--fp16 \
--video_encoder_type 'videoswin' \
--videoswin_timestride 1 \
--checkpoint $checkpoint &

### finetune on msrvtt fast MCQA
CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python ./train_fast_retrieval.py \
--config ./config/MCQA-fast-msrvtt-b32-2gpu-lr5e-5-1witer.json \
--output_dir $basedir'/MCQA-fast-msrvtt'  \
--fp16 \
--video_encoder_type 'videoswin' \
--videoswin_timestride 1 \
--checkpoint $checkpoint &

# CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./train_slow_retrieval.py \
# --config ./config/slow-retrieval-msrvtt-4gpu-b8-lr5e-5-3witer.json \
# --output_dir $basedir'/slow-retrieval-msrvtt'  \
# --fp16 \
# --video_encoder_type 'videoswin' \
# --checkpoint $checkpoint &











# ### finetune on msrvtt slow MCQA
# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python ./train_slow_retrieval.py \
# --config ./config/MCQA-slow-msrvtt-b64-lr1e-5-6witer-4-224-32.json \
# --output_dir $basedir'/MCQA-slow-msrvtt' \
# --fp16 \
# --checkpoint "./output/pretrianed_weights/ima21k_vit_base_224_32.pt" &