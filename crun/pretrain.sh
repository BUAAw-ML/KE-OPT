CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-webvid-2modal-b96-4gpu-4w-1-224-16-mlm+contra+unimlm-fp16-lr1e-4-timesformer-space-12-12-3.json \
--output_dir ./output/v2.0-final-pretrain-webvid-2modal-b96-4gpu-4w-1-224-16-mlm+contra+unimlm-fp16-lr1e-4-timesformer-space-12-12-3 \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero.pt" \


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python ./pretrain.py \
--config ./config/pretrain-webvid-2modal-b42-8gpu-4.5w-4-224-16-mlm+contra+unimlm-fp16-lr1e-4-timesformer-space-12-12-3.json \
--output_dir ./output/v2.0-final-pretrain-webvid-2modal-b42-8gpu-4.5w-4-224-16-mlm+contra+unimlm-fp16-lr1e-4-timesformer-space-12-12-3 \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero.pt" \



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python ./pretrain.py \
--config ./config/pretrain-webvid+cc3m-2modal-b96-8gpu-4.5w-1-224-16-mlm+contra+unimlm-fp16-lr1e-4-timesformer-space-12-12-3.json \
--output_dir ./output/v2.0-final-pretrain-webvid+cc3m-2modal-b96-8gpu-4.5w-1-224-16-mlm+contra+unimlm-fp16-lr1e-4-timesformer-space-12-12-3 \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero.pt" \

CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-cc3m-2modal-b96-4gpu-5w-1-224-16-mlm+contra+unimlm-fp16-lr1e-4-timesformer-space-12-12-3.json \
--output_dir ./output/v2.0-final-pretrain-cc3m-2modal-b96-4gpu-5w-1-224-16-mlm+contra+unimlm-fp16-lr1e-4-timesformer-space-12-12-3 \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero.pt" \


##########
CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-webvid-2modal-b96-4gpu-5.5w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm.json \
--output_dir ./output/v2.0-final-pretrain-webvid-2modal-b96-4gpu-5.5w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero.pt" \

CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-webvid-2modal-b96-4gpu-5.5w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-8-12-4-prenorm.json \
--output_dir ./output/v2.0-final-pretrain-webvid-2modal-b96-4gpu-5.5w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-8-12-4-prenorm \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero_8_12_4.pt" \

CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-webvid-2modal-b96-4gpu-5.5w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-8-12-4-prenorm-woreuse.json \
--output_dir ./output/v2.0-final-pretrain-webvid-2modal-b96-4gpu-5.5w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-8-12-4-prenorm-woreuse \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero_8_12_4.pt" \

CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-webvid-2modal-b96-4gpu-5.5w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-8-12-4-postnorm-woreuse.json \
--output_dir ./output/v2.0-final-pretrain-webvid-2modal-b96-4gpu-5.5w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-8-12-4-postnorm-woreuse \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero_8_12_4.pt" \
####


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python ./pretrain.py \
--config ./config/pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm.json \
--output_dir ./output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero.pt" \


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python ./pretrain.py \
--config ./config/pretrain-webvid-2modal-b42-8gpu-6w-4-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3.json \
--output_dir ./output/v2.0-final-pretrain-webvid-2modal-b42-8gpu-6w-4-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3 \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero.pt" \



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python ./pretrain.py \
--config ./config/pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch.json \
--output_dir ./output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-alternatebatch \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero.pt" \


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python ./pretrain.py \
--config ./config/pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-resizecrop.json \
--output_dir ./output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-resizecrop \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero.pt" \






CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python ./pretrain.py \
--config ./config/pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm.json \
--output_dir ./output/v2.0-final-pretrain-webvid+cc3m-2modal-b92-8gpu-6w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-wd0.001 \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero.pt" \




CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-webvid-2modal-b96-4gpu-5.5w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm.json \
--output_dir ./output/v2.0-final-pretrain-webvid-2modal-b96-4gpu-5.5w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-wd0.001 \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero.pt" \

CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-webvid-2modal-b96-4gpu-5.5w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-contrafilter.json \
--output_dir ./output/v2.0-final-pretrain-webvid-2modal-b96-4gpu-5.5w-1-224-16-mlm+contra+unimlm+match-fp16-lr1e-4-timesformer-space-12-12-3-prenorm-contrafilter \
--checkpoint "./output/pretrianed_weights/double_modify_timesformerbase16_bertbaseuncased_timezero.pt" \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python ./pretrain.py \
--config ./config/pretrain-webvid+cc3m-2modal-b128-8gpu-5w-1-224-16-spaceaverage.json \
--output_dir ./output/v3.0_start_pretrain-webvid+cc3m-2modal-b128-8gpu-5w-1-224-16-spaceaverage  



CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo  

CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwofalse.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwofalse 


