CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_caption.py \
--config ./config/caption-msrvtt-b64-lr5e-5-2witer.json \
--output_dir ./output/caption-msrvtt-b64-lr5e-5-2witer \



CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_caption.py \
--config ./config/caption-msrvtt-b64-lr5e-5-2witer-maskprob0.6.json \
--output_dir ./output/caption-msrvtt-b64-lr5e-5-2witer-maskprob0.6 \

CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_caption.py \
--config ./config/caption-msrvtt-b64-lr5e-5-2witer-maskprob0.6.json \
--output_dir ./output/caption-msrvtt-b64-lr5e-5-2witer-maskprob0.6 \
--zero_shot \
--checkpoint "/home/zhongguokexueyuanzidonghuayanjiusuo/shchen/videoOPT-Three/output/caption-msrvtt-b64-lr5e-5-2witer-maskprob0.6/ckpt/model_step_20000.pt"