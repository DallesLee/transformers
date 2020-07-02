dataset="MRPC"
task="${dataset~~}"

python run_pruning.py \
	--data_dir glue_data/$dataset/ \
	--model_name_or_path output/$dataset/ \
	--task_name $task \
	--output_dir pruning_output/$dataset/ \
	--cache_dir cache/$dataset/ \
	--tokenizer_name bert-base-uncased \