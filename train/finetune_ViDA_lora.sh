#!/bin/bash
MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
MODEL_LOCAL_PATH=/path/to/your/local/model

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=8
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
MASTER_PORT=34229

DATA_PATH=../ViDA-MIPI-dataset/train/ViDA-qwen2/combined.json
EVAL_DATA_PATH=
IMAGE_FOLDER=../ViDA-MIPI-dataset/train/images

OUTPUT_DIR=/path/to/your/output/dir
export PYTHONPATH=src:$PYTHONPATH

# If you have local model, replace value of model_id from MODEL_NAME to MODEL_LOCAL_PATH
deepspeed \
    --master_port $MASTER_PORT \
    src/training/train.py \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((8192 * 28 * 28)) \
    --learning_rate 1e-4 \
    --merger_lr 1e-5 \
    --vision_lr 1e-6 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --dataloader_num_workers 4 \