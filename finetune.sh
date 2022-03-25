basedir=$1


# CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-224-16-5e-5-bs64-gpu1-90000-augnone-maskprob0.6-labelsmoothing0.1'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 64 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --video_aug 'woaug' \
# --label_smoothing 0.1 \
# --txt_mask_prob 0.6 \
# --pretrain_dir $basedir \
# --fp16  


# CUDA_VISIBLE_DEVICES=5,6 horovodrun -np 2 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-384-16-2e-5-bs32-gpu2-90000-augnone-maskprob0.6-labelsmoothing0.1'  \
# --resolution 384 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 2e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --video_aug 'woaug' \
# --label_smoothing 0.1 \
# --txt_mask_prob 0.6 \
# --pretrain_dir $basedir \
# --fp16  &

# CUDA_VISIBLE_DEVICES=5,6 horovodrun -np 2 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-384-16-5e-5-bs32-gpu2-90000-augnone-maskprob0.6-labelsmoothing0.1'  \
# --resolution 384 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --video_aug 'woaug' \
# --label_smoothing 0.1 \
# --txt_mask_prob 0.6 \
# --pretrain_dir $basedir \
# --fp16  &

# CUDA_VISIBLE_DEVICES=5,6 horovodrun -np 2 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-384-16-1e-4-bs32-gpu2-90000-augnone-maskprob0.6-labelsmoothing0.1'  \
# --resolution 384 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 1e-4 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --video_aug 'woaug' \
# --label_smoothing 0.1 \
# --txt_mask_prob 0.6 \
# --pretrain_dir $basedir \
# --fp16  

# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-384-16-5e-5-bs32-gpu4-90000-augnone-maskprob0.6-labelsmoothing0.1'  \
# --resolution 384 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --video_aug 'woaug' \
# --label_smoothing 0.1 \
# --txt_mask_prob 0.6 \
# --pretrain_dir $basedir \
# --fp16  &


# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-4-224-16-lr2e-5-bs32-gpu2-step20000-woaudio'   \
# --sample_frame 4 \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 2e-5 \
# --num_train_steps 20000 \
# --use_audio false \
# --pretrain_dir $basedir 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-8-224-16-lr2e-5-bs32-gpu2-step20000-woaudio'   \
# --sample_frame 8 \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 2e-5 \
# --num_train_steps 20000 \
# --use_audio true \
# --zero_shot \
# --video_aug 'woaug' \
# --checkpoint "/raid/shchen/videoOPT-Three/output/v3.0_start_pretrain-webvid+cc3m+audioset-3modal-7w-1:1:1-1-224-16-spaceaverage-strengthentwo/openended-vqa-msrvtt-fps4_8-224-16-lr2e-5-bs8-gpu8-step20000-wd0.001/ckpt/model_step_20000.pt" \
# --pretrain_dir $basedir 


# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python ./train_caption.py \
# --config ./config/caption-msrvtt.json \
# --output_dir $basedir'/caption-msrvtt-4-224-16-lr2e-5-bs32-gpu2-step20000-woaudio'   \
# --sample_frame 4 \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 2e-5 \
# --num_train_steps 20000 \
# --use_audio false \
# --pretrain_dir $basedir &

# ## finetune on msrvtt open-ended vqa
# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt.json \
# --output_dir $basedir'/openended-vqa-msrvtt-4-224-16-lr2e-5-bs32-gpu2-step20000-woaudio'   \
# --sample_frame 4 \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 2e-5 \
# --num_train_steps 20000 \
# --use_audio false \
# --pretrain_dir $basedir 



# CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-224-16-2e-5-bs64-gpu1-90000'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 64 \
# --learning_rate 2e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --pretrain_dir $basedir \
# --fp16  & 

# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-384-16-2e-5-bs32-gpu2-90000'  \
# --resolution 384 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 2e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --pretrain_dir $basedir \
# --fp16  



# CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-coco.json \
# --output_dir $basedir'/fast-retrieval-coco-224-16-2e-5-bs64-gpu1-90000'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 64 \
# --learning_rate 2e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --pretrain_dir $basedir \
# --fp16  & 

# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-coco.json \
# --output_dir $basedir'/fast-retrieval-coco-384-16-2e-5-bs32-gpu2-90000'  \
# --resolution 384 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 2e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --pretrain_dir $basedir \
# --fp16  




# CUDA_VISIBLE_DEVICES=4,5,6,7  horovodrun -np 4 python ./train_image_classification.py \
# --config ./config/image-classification-imagenet.json \
# --output_dir $basedir'/image-classification-imagenet-224-16-1e-3-bs256-gpu4-38000'  \
# --pretrain_dir $basedir \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 248 \
# --learning_rate 1e-3 \
# --num_train_steps 38000 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --fp16  


# CUDA_VISIBLE_DEVICES=4,5,6,7  horovodrun -np 4 python ./train_image_classification.py \
# --config ./config/image-classification-imagenet.json \
# --output_dir $basedir'/image-classification-imagenet-224-16-2e-5-bs256-gpu4-38000'  \
# --pretrain_dir $basedir \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 248 \
# --learning_rate 2e-5 \
# --num_train_steps 38000 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --fp16  