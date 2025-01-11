#!/bin/bash

MODEL_NAME="HuggingFaceTB/SmolVLM-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /home/workspace/Qwen2-VL-Finetune/output/testing_lora \
    --model-base $MODEL_NAME  \
    --save-model-path /home/workspace/Qwen2-VL-Finetune/output/merge_test \
    --safe-serialization