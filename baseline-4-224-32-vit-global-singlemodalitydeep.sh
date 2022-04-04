basedir="./output/v2.0_double_modify_new_baseline_4_224_32_vit_global_singlemodality_deep"
checkpoint='./output/pretrianed_weights/double_modify_vitbase32_bertbaseuncased.pt'

### finetune on msrvtt fast retrieval
CUDA_VISIBLE_DEVICES=6 horovodrun -np 1 python ./train_fast_retrieval.py \
--config ./config/fast-retrieval-msrvtt-b64-lr5e-5-2witer.json \
--output_dir $basedir'/fast-retrieval-msrvtt'  \
--fp16  \
--video_encoder_type 'vit_global' \
--singlemodality_shallow false \
--checkpoint $checkpoint  &




### finetune on msrvtt fast MCQA
CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python ./train_fast_retrieval.py \
--config ./config/MCQA-fast-msrvtt-b64-lr5e-5-1witer.json \
--output_dir $basedir'/MCQA-fast-msrvtt'  \
--fp16 \
--video_encoder_type 'vit_global' \
--singlemodality_shallow false \
--checkpoint $checkpoint &

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