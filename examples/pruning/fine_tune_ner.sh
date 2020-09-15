#!/bin/sh
# Prepare dataset
data_dir="data/NER"

mkdir -p $data_dir
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-train.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > $data_dir/train.txt.tmp
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-dev.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > $data_dir/dev.txt.tmp
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-test.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > $data_dir/test.txt.tmp

python ../token-classification/scripts/preprocess.py $data_dir/train.txt.tmp bert-base-multilingual-cased 128 > $data_dir/train.txt
python ../token-classification/scripts/preprocess.py $data_dir/dev.txt.tmp bert-base-multilingual-cased 128 > $data_dir/dev.txt
python ../token-classification/scripts/preprocess.py $data_dir/test.txt.tmp bert-base-multilingual-cased 128 > $data_dir/test.txt

cat $data_dir/train.txt $data_dir/dev.txt $data_dir/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $data_dir/labels.txt

# Fine-tune
python ../token-classification/run_ner.py --data_dir $data_dir \
--labels $data_dir/labels.txt \
--model_name_or_path bert-base-multilingual-cased \
--output_dir output/NER/ \
--cache_dir cache/NER/ \
--max_seq_length  128 \
--num_train_epochs 3 \
--per_device_train_batch_size 32 \
--save_steps 750 \
--seed 1 \
--do_train \
--do_eval \