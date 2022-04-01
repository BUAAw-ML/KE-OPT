CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./pretrain.py \
--config ./config/pretrain-cc3m-2modal-b128-4gpu-5w-1-224-16-spaceaverage.json \
--output_dir ./output/retrain-cc3m-2modal-b128-4gpu-5w-1-224-16-spaceaverage