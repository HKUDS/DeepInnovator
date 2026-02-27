#!/bin/bash

IF_TEST=False
# True

IF_L1=False
#True
IF_L0=False
#True
IF_L2=True
#True
#False
OUTPUT_DIR="./data/train"

# Option 1: Process local data using preprocess.py
python3 recipe/DeepInnovator/preprocess.py \
    --input_dir "./data/arxiv_data" \
    --task_desc "refine a research idea" \
    --output_dir $OUTPUT_DIR \
    --validation_size 0.1 \
    --seed 42 \
    --num_proc 1 \
    --dataset_type "rl" \
    --test $IF_TEST \
    --layer1 $IF_L1 \
    --layer0 $IF_L0 \
    --layer2 $IF_L2

# Option 2: Download and convert HuggingFace dataset to training format using HF2local.py
# This script downloads data from HuggingFace and converts it to the format ready for training
# Uncomment the following lines and set the HF_DATASET variable to use this option:
#
# HF_DATASET="username/dataset_name"  # Replace with your HuggingFace dataset path
# python3 recipe/DeepInnovator/HF2local.py \
#     --dataset $HF_DATASET \
#     --train_split "train" \
#     --val_split "validation" \
#     --output_dir $OUTPUT_DIR \
#     --task_desc "refine a research idea"

#