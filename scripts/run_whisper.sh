#!/bin/bash

# ./run_lora_vs_nonlora.sh
GPU=0
WANDB_PROJ=loraxs

exec 2> logs/${WANDB_PROJ}_error.txt 1> logs/${WANDB_PROJ}_log.txt

# LORA-XS WITHOUT DROPOUT:
python main_whisper.py \
        --model_id openai/whisper-large-v3 \
        --wandb_project $WANDB_PROJ \
        --gpu $GPU \
        --optimizer True \
        --scheduler linear \
        --batch_size 2 \
        --accumulation_batch 64 \
        --lora_r 16 \
        --lora_alpha 64 \
        --n_epochs 3 \
        --lora_init lora \
        --run_name loraxs-large-v3 \
        --target_modules all \
        --lora_dropout 0.0 \
