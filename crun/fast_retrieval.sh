CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_fast_retrieval.py \
--config ./config/fast-retrieval-msrvtt-b64-lr5e-5-6witer.json \
--output_dir ./output/fast-retrieval-msrvtt-b64-lr5e-5-6witer


CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_fast_retrieval.py \
--config ./config/fast-retrieval-msrvtt-b64-lr5e-5-6witer-dualsoftmax.json \
--output_dir ./output/fast-retrieval-msrvtt-b64-lr5e-5-6witer-dualsoftmax



CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python ./train_fast_retrieval.py \
--config ./config/fast-retrieval-msrvtt-b64-lr5e-5-6witer-1-224-16.json \
--output_dir ./output/fast-retrieval-msrvtt-b64-lr5e-5-6witer-1-224-16 \
--checkpoint "/home/zhongguokexueyuanzidonghuayanjiusuo/shchen/videoOPT-Three/output/v2.0-pretrain-webvid-2modal-b150-4gpu-8w-1-224-16-unimlm-contra-fp16-vitpretrained/ckpt/model_step_80000.pt" \
--zero_shot 

