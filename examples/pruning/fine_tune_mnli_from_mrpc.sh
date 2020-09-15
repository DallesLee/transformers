#!/bin/sh
dataset="MNLI"
task="${dataset~~}"

# python ../../utils/download_glue_data.py --data_dir data/


python ../text-classification/run_glue.py \
	--model_name_or_path output/MRPC/ \
	--task_name $dataset \
	--do_train \
	--do_eval \
	--data_dir data/$dataset/ \
	--max_seq_length 128 \
	--per_gpu_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 3.0 \
	--output_dir output/${dataset}_MRPC/ \
	--cache_dir cache/${dataset}_MRPC/ \
	--save_steps 10000 \