dataset="NER"
task="${dataset~~}"

python prune_ner.py \
	--data_dir data/$dataset/ \
	--labels data/$dataset/labels.txt \
	--model_name_or_path output/$dataset/ \
	--task_name $task \
	--output_dir pruning_output/$dataset/ \
	--cache_dir cache/$dataset/ \
	--tokenizer_name bert-base-multilingual-cased \