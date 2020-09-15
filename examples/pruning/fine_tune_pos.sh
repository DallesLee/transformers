#!/bin/sh
# Prepare dataset
data_dir="data/POS"

# Fine-tune
python ../token-classification/run_ner.py --data_dir $data_dir \
--labels $data_dir/labels.txt \
--model_name_or_path bert-base-cased \
--output_dir output/POS/ \
--cache_dir cache/POS/ \
--max_seq_length  135 \
--num_train_epochs 3 \
--per_device_train_batch_size 32 \
--save_steps 10000 \
--seed 1 \
--do_train \
--do_eval \