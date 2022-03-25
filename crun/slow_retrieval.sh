CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_slow_retrieval.py \
--config ./config/slow-retrieval-msrvtt-b64-lr5e-5-6witer-warmup0.1.json \
--output_dir ./output/slow-retrieval-msrvtt-b64-lr5e-5-6witer-warmup0.1

CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_slow_retrieval.py \
--config ./config/slow-retrieval-msrvtt-b64-lr1e-5-6witer.json \
--output_dir ./output/slow-retrieval-msrvtt-b64-lr1e-5-6witer \
--checkpoint /home/zhongguokexueyuanzidonghuayanjiusuo/shchen/videoOPT-Three/output/slow-retrieval-msrvtt-b64-lr1e-5-6witer/ckpt/model_step_60000.pt \
--zero_shot












CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_slow_retrieval_contra.py \
--config ./config/slow-retrieval-contra-msrvtt-b16-lr5e-5-6witer.json \
--output_dir ./output/slow-retrieval-contra-msrvtt-b16-lr5e-5-6witer



CUDA_VISIBLE_DEVICES=1,2 horovodrun -np 2 python ./train_slow_retrieval_contra.py \
--config ./config/slow-retrieval-contra-msrvtt-b16-lr5e-5-6witer.json \
--output_dir ./output/slow-retrieval-contra-msrvtt-b16-lr5e-5-6witer-2gpu

CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./train_slow_retrieval_contra.py \
--config ./config/slow-retrieval-contra-msrvtt-b16-lr5e-5-6witer.json \
--output_dir ./output/slow-retrieval-contra-msrvtt-b16-lr5e-5-6witer-4gpu


CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./train_slow_retrieval_contra.py \
--config ./config/slow-retrieval-contra-msrvtt-b16-lr5e-5-6witer.json \
--output_dir ./output/slow-retrieval-contra-msrvtt-b16-lr5e-5-6witer-4gpu-memorytest


CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./train_slow_retrieval_contra_metaloader.py \
--config ./config/slow-retrieval-contra-msrvtt-b16-lr5e-5-6witer.json \
--output_dir ./output/slow-retrieval-contra-msrvtt-b16-lr5e-5-6witer-4gpu-memorytest



CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_slow_retrieval.py \
--config ./config/slow-retrieval-msrvtt-b64-lr5e-5-6witer.json \
--output_dir ./output/slow-retrieval-msrvtt-b64-lr5e-5-6witer

CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_slow_retrieval.py \
--config ./config/slow-retrieval-msrvtt-b64-lr5e-5-6witer-warmup0.1.json \
--output_dir ./output/slow-retrieval-msrvtt-b64-lr5e-5-6witer-warmup0.1-ft \
--checkpoint "/home/zhongguokexueyuanzidonghuayanjiusuo/shchen/videoOPT-Three/output/pretrain-webvid-2modal-b250-4gpu-20w-mlm-match-contra-unimlm/ckpt/model_step_200000.pt"

