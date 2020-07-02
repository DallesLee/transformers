#!/bin/sh
# python ../../utils/download_glue_data.py --data_dir glue_data/

python ../text-classification/run_glue.py \
	--model_name_or_path bert-base-uncased \
	--task_name MRPC \
	--do_train \
	--do_eval \
	--data_dir glue_data/MRPC/ \
	--max_seq_length 128 \
	--per_gpu_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 3.0 \
	--output_dir output/ \
	--cache_dir cache/ \
	--save_steps 10000 \