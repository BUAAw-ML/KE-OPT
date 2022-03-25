### open-ended


CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_open_ended_vqa.py \
--config ./config/openendedvqa-msrvtt-b64-lr5e-5-6witer-answer1500.json \
--output_dir ./output/openendedvqa-msrvtt-b64-lr5e-5-6witer-answer1500 \


CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_open_ended_vqa.py \
--config ./config/openendedvqa-msrvtt-b64-lr5e-5-6witer-answerfull.json \
--output_dir ./output/openendedvqa-msrvtt-b64-lr5e-5-6witer-answerfull \


CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_open_ended_vqa.py \
--config ./config/openendedvqa-msrvtt-b128-lr5e-5-6witer-answer1500.json \
--output_dir ./output/openendedvqa-msrvtt-b128-lr5e-5-6witer-answer1500 \


CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_open_ended_vqa.py \
--config ./config/openendedvqa-msrvtt-b64-lr5e-5-2witer-answerfull.json \
--output_dir ./output/openendedvqa-msrvtt-b64-lr5e-5-2witer-answerfull \


CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_open_ended_vqa.py \
--config ./config/openendedvqa-msrvtt-b64-lr5e-5-14kiter-answerfull.json \
--output_dir ./output/openendedvqa-msrvtt-b64-lr5e-5-14kiter-answerfull-ft \
--checkpoint "/home/zhongguokexueyuanzidonghuayanjiusuo/shchen/videoOPT-Three/output/pretrain-webvid-2modal-b250-4gpu-20w-mlm-match-contra-unimlm/ckpt/model_step_200000.pt"


####multiple-choice
CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_fast_retrieval.py \
--config ./config/MCQA-fast-msrvtt-b64-lr5e-5-6witer.json \
--output_dir ./output/MCQA-fast-msrvtt-b64-lr5e-5-6witer \


CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_fast_retrieval.py \
--config ./config/MCQA-fast-msrvtt-b64-lr5e-5-2witer.json \
--output_dir ./output/MCQA-fast-msrvtt-b64-lr5e-5-2witer-ft \
--checkpoint "/home/zhongguokexueyuanzidonghuayanjiusuo/shchen/videoOPT-Three/output/pretrain-webvid-2modal-b250-4gpu-20w-mlm-match-contra-unimlm/ckpt/model_step_200000.pt"




CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./train_slow_retrieval_contra.py \
--config ./config/MCQA-slow-contra-msrvtt-b16-lr5e-5-3witer.json \
--output_dir ./output/MCQA-slow-contra-msrvtt-b16-lr5e-5-3witer-ft \
--checkpoint "/home/zhongguokexueyuanzidonghuayanjiusuo/shchen/videoOPT-Three/output/pretrain-webvid-2modal-b250-4gpu-20w-mlm-match-contra-unimlm/ckpt/model_step_200000.pt"



CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./train_slow_retrieval_contra.py \
--config ./config/MCQA-slow-contra-msrvtt-b16-lr5e-5-2witer.json \
--output_dir ./output/MCQA-slow-contra-msrvtt-b16-lr5e-5-2witer \

CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python ./train_slow_retrieval.py \
--config ./config/MCQA-slow-msrvtt-b64-lr1e-5-6witer.json \
--output_dir ./output/MCQA-slow-msrvtt-b64-lr1e-5-6witer \




CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python ./train_slow_retrieval.py \
--config ./config/MCQA-slow-msrvtt-b64-lr5e-5-6witer-warmup0.1.json \
--output_dir ./output/MCQA-slow-msrvtt-b64-lr5e-5-6witer-warmup0.1 \



CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python ./train_slow_retrieval.py \
--config ./config/MCQA-slow-msrvtt-b64-lr5e-5-6witer-warmup0.1.json \
--output_dir ./output/MCQA-slow-msrvtt-b64-lr5e-5-6witer-warmup0.1-ft \
--checkpoint "/home/zhongguokexueyuanzidonghuayanjiusuo/shchen/videoOPT-Three/output/pretrain-webvid-2modal-b250-4gpu-20w-mlm-match-contra-unimlm/ckpt/model_step_200000.pt"