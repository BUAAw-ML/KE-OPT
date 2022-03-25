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

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python ./pretrain.py \
--config ./config/pretrain-webvid+cc3m+audioset-3modal-7w-1:1:1-1-224-16-spaceaverage-strengthentwo.json \
--output_dir ./output/v3.0_start_pretrain-webvid+cc3m+audioset-3modal-7w-1:1:1-1-224-16-spaceaverage-strengthentwo





CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-mvmfeatregerssion.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-mvmfeatregerssion


CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-mvmpixelclassification.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-mvmpixelclassification-new


CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-mvmpixelregerssion.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-mvmpixelregerssion
--only_eval
--checkpoint "/raid/shchen/videoOPT-Three/output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-mvmpixelregerssion/"


CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-featclassification.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-featclassification



CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-mvmfeatregerssion-visualvqvae.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-mvmfeatregerssion-visualvqvae




CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-visualvqvae.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-visualvqvae





CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-mvmfeatregerssion.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-mvmfeatregerssion_blockmasing





CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-attentivemasktxt_inter.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-attentivemasktxt_inter


CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-attentivemasktxt_intra.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-attentivemasktxt_intra


CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-matchThree.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-matchThree


CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 1 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-mvmfeatregerssion-clip.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-mvmfeatregerssion-clip




CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-cc3m-2modal-b128-4gpu-5.5w-1-224-16-spaceaverage.json \
--output_dir ./output/v3.0_start_pretrain-cc3m-2modal-b128-4gpu-5.5w-1-224-16-spaceaverage



CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-cc3m-2modal-b128-4gpu-5.5w-1-224-16-spaceaverage-woaug.json \
--output_dir ./output/v3.0_start_pretrain-cc3m-2modal-b128-4gpu-5.5w-1-224-16-spaceaverage-woaug


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python ./pretrain.py \
--config ./config/pretrain-cc3m-2modal-b44-8gpu-8.4w-1-384-16-spaceaverage-woaug.json \
--output_dir ./output/v3.0_start_pretrain-cc3m-2modal-b44-8gpu-8.4w-1-384-16-spaceaverage-woaug


