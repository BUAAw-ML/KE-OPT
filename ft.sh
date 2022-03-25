CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-attentivemasktxt_inter.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-attentivemasktxt_inter 


sh auto_finetune-4-224-16-timesformer-space-2e-5-64-2w-4fps-3m-useaudio-0123.sh \
./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-attentivemasktxt_inter 


CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-attentivemasktxt_intra.json \
--output_dir ./output/v3.0_start_pretrain-audioset-3modal-b64-4gpu-4w-1-224-16-spaceaverage-strengthentwo-attentivemasktxt_intra

