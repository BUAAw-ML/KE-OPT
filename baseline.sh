basedir='./output/v3.0-baseline-4-224-16-timesformer-space-coco'



# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-4-224-16-lr5e-5-bs32-gpu2-step20000-woaudio'   \
# --sample_frame 4 \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 5e-5 \
# --num_train_steps 20000 \
# --use_audio false &


# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python ./train_caption.py \
# --config ./config/caption-msrvtt.json \
# --output_dir $basedir'/caption-msrvtt-4-224-16-lr5e-5-bs32-gpu2-step20000-woaudio'   \
# --sample_frame 4 \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 5e-5 \
# --num_train_steps 20000 \
# --use_audio false &

# ## finetune on msrvtt open-ended vqa
# CUDA_VISIBLE_DEVICES=4,5 horovodrun -np 2 python ./train_open_ended_vqa.py \
# --config ./config/VQA-msrvtt.json \
# --output_dir $basedir'/openended-vqa-msrvtt-4-224-16-lr5e-5-bs32-gpu2-step20000-woaudio'   \
# --sample_frame 4 \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 5e-5 \
# --num_train_steps 20000 \
# --use_audio false 

# CUDA_VISIBLE_DEVICES=0,1,2,3  horovodrun -np 4 python ./train_image_classification.py \
# --config ./config/image-classification-imagenet.json \
# --output_dir $basedir'/image-classification-imagenet-224-16-1e-3-bs256-gpu4-38000'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 248 \
# --learning_rate 1e-3 \
# --num_train_steps 38000 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --fp16  


# CUDA_VISIBLE_DEVICES=0,1,2,3  horovodrun -np 4 python ./train_image_classification.py \
# --config ./config/image-classification-imagenet.json \
# --output_dir $basedir'/image-classification-imagenet-224-16-2e-5-bs256-gpu4-38000'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 248 \
# --learning_rate 2e-5 \
# --num_train_steps 38000 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --fp16  

# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-224-16-5e-5-bs64-gpu1-90000-maskprob0.6'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 64 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --txt_mask_prob 0.6 \
# --fp16  

# CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-224-16-5e-5-bs128-gpu1-90000'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 128 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --fp16  &

# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-224-16-5e-5-bs64-gpu1-90000-augnone-maskprob0.6'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 64 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --video_aug 'woaug' \
# --txt_mask_prob 0.6 \
# --fp16  &

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
# --fp16  &

# CUDA_VISIBLE_DEVICES=3,4 horovodrun -np 2 python ./train_caption.py \
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
# --fp16  





# CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-224-16-5e-5-bs128-gpu1-90000-augnone'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 128 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --video_aug 'woaug' \
# --checkpoint /raid/shchen/videoOPT-Three/output/v3.0-baseline-4-224-16-timesformer-space-coco/caption-coco-224-16-5e-5-bs64-gpu1-90000-augnone/ckpt/model_step_90000.pt \
# --zero_shot \
# --fp16  

# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-224-16-5e-5-bs64-gpu1-180000-augnone-maskprob0.6'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 64 \
# --learning_rate 5e-5 \
# --num_train_steps 180000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --video_aug 'woaug' \
# --txt_mask_prob 0.6 \
# --fp16  


# CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-224-16-5e-5-bs64-gpu1-90000-augufo'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 64 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --video_aug 'randomresizedcrop_and_flip_ufo' \
# --checkpoint /raid/shchen/videoOPT-Three/output/v3.0-baseline-4-224-16-timesformer-space-coco/caption-coco-224-16-5e-5-bs64-gpu1-90000-augufo/ckpt/model_step_90000.pt \
# --zero_shot \
# --fp16  

# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-224-16-5e-5-bs64-gpu1-90000-augrand'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 64 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --video_aug 'randaug' \
# --fp16  


# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-224-16-5e-5-bs64-gpu1-90000-labelsmoothing0.1'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 64 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --label_smoothing 0.1 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --fp16  


# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-224-16-5e-5-bs64-gpu1-90000'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 64 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --fp16  &



# CUDA_VISIBLE_DEVICES=1,2 horovodrun -np 2 python ./train_caption.py \
# --config ./config/caption-coco.json \
# --output_dir $basedir'/caption-coco-384-16-5e-5-bs32-gpu2-90000'  \
# --resolution 384 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --fp16  




# CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-coco.json \
# --output_dir $basedir'/fast-retrieval-coco-224-16-5e-5-bs64-gpu1-90000-woaug'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 64 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_aug 'woaug' \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --fp16  &


# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-coco.json \
# --output_dir $basedir'/fast-retrieval-coco-224-16-5e-5-bs128-gpu1-90000-woaug'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 128 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --video_aug 'woaug' \
# --use_audio false \
# --fp16  &


CUDA_VISIBLE_DEVICES=4,7 horovodrun -np 2 python ./train_fast_retrieval.py \
--config ./config/fast-retrieval-coco.json \
--output_dir $basedir'/fast-retrieval-coco-224-16-1e-4-bs256-gpu1-46000-woaug'  \
--resolution 224 \
--patch_size 16 \
--train_batch_size 128 \
--learning_rate 1e-4 \
--num_train_steps 46000 \
--weight_decay 0.001 \
--video_encoder_type 'timesformer_space' \
--use_audio false \
--video_aug 'woaug' \
--fp16  

# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-coco.json \
# --output_dir $basedir'/fast-retrieval-coco-224-16-5e-5-bs64-gpu1-90000'  \
# --resolution 224 \
# --patch_size 16 \
# --train_batch_size 64 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --fp16  






# CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-coco.json \
# --output_dir $basedir'/fast-retrieval-coco-384-16-5e-5-bs32-gpu2-90000'  \
# --resolution 384 \
# --patch_size 16 \
# --train_batch_size 32 \
# --learning_rate 5e-5 \
# --num_train_steps 90000 \
# --weight_decay 0.001 \
# --video_encoder_type 'timesformer_space' \
# --use_audio false \
# --fp16  















# #finetune on msrvtt fast retrieval
# CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np $ngpu python ./train_fast_retrieval.py \
# --config ./config/fast-retrieval-msrvtt.json \
# --output_dir $basedir'/fast-retrieval-msrvtt-'$prefix$fps_or_frames'_'$sample_num'-'$resolution'-'$patch_size'-lr'$learning_rate'-bs'$train_batch_size'-gpu'$ngpu'-step'$num_train_steps'-wd'$weight_decay'-useaudio'$use_audio   \
# --checkpoint $checkpoint \
# --video_path $video_path  \
# --sample_frame $sample_num \
# --resolution $resolution \
# --patch_size $patch_size \
# --train_batch_size $train_batch_size \
# --learning_rate $learning_rate \
# --num_train_steps $num_train_steps \
# --weight_decay $weight_decay \
# --video_encoder_type $video_encoder_type \
# --use_audio $use_audio \
# --fp16   &
