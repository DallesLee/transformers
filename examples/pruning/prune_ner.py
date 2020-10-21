import argparse
import logging
import os
from datetime import datetime
# import sampling_utils

import matplotlib.pylab as plt
from sklearn.metrics import auc
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)

import sys
sys.path.append("../token-classification/")
from utils_ner import NerDataset, Split, get_labels
from typing import Dict

from pruning_utils import *

logger = logging.getLogger(__name__)

def evaluate(args, model, eval_dataloader, head_mask=None):
    if args.n_gpu > 1:
        n_layers, n_heads = model.module.config.num_hidden_layers, model.module.config.num_attention_heads
    else:
        n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(args.device)

    # Evaluate
    preds = None
    labels = None
    for step, inputs in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(**inputs, head_mask=head_mask)
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)
    
    # preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    score = compute_metrics(EvalPrediction(predictions=preds, label_ids=labels))['f1']
        
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        type=str,
        required=True,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--data_subset", type=int, default=-1, help="If > 0: limit the data to a subset of data_subset instances."
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Whether to overwrite data in output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--exact_pruning", action="store_true", help="Compute head importance for each step"
    )
    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )
    parser.add_argument(
        "--dont_use_abs", action="store_true", help="Don't apply abs on first order derivative"
    )
    parser.add_argument(
        "--use_second", action="store_true", help="Use second order derivative as quality"
    )
    parser.add_argument(
        "--use_squared", action="store_true", help="Use squared derivative as quality"
    )
    parser.add_argument(
        "--use_contexts", action="store_true", help="Use context vectors instead of attentions weights"
    )

    parser.add_argument(
        "--masking_amount", default=0.1, type=float, help="Amount to heads to masking at each masking step."
    )
    parser.add_argument("--metric_name", default="acc", type=str, help="Metric to use for head masking.")

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, sequences shorter padded.",
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup devices and distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend

    # Setup logging
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed: {}".format(args.device, args.n_gpu, bool(args.local_rank != -1)))

    # Set seeds
    set_seed(args.seed)

    # Prepare CONLL-2003 task
    args.task_name = args.task_name.lower()
    labels = get_labels(args.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=False,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    model.eval()
    # Distributed and parallel training
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Print/save training arguments
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(args, os.path.join(args.output_dir, "run_args.bin"))
    logger.info("Training/evaluation parameters %s", args)

    # Get datasets
    train_dataset = NerDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=args.max_seq_length,
            overwrite_cache=False,
            mode=Split.train,
        )
    split = int(len(train_dataset) * 0.9)
    train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=default_data_collator
    )
    val_dataset = Subset(train_dataset, list(range(split, len(train_dataset))))
    val_sampler = SequentialSampler(val_dataset) if args.local_rank == -1 else DistributedSampler(val_dataset)
    val_dataloader = DataLoader(
        val_dataset, sampler=val_sampler, batch_size=args.batch_size, collate_fn=default_data_collator
    )
    eval_dataset = NerDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=args.max_seq_length,
            overwrite_cache=args.overwrite_cache,
            mode=Split.dev,
        )
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=default_data_collator
    )

    args.exact_pruning = True
    args.dont_normalize_importance_by_layer = True
    scores, sparsities, all_head_masks = mask_heads(
        args, model, val_dataloader, eval_dataloader
    )
    logger.info("Area under curve: %.2f", auc(sparsities, scores))
    
    # scores, sparsities, all_head_masks = unmask_heads(
    #     args, model, train_dataloader, eval_dataloader
    # )
    # logger.info("Area under curve: %.2f", auc(sparsities, scores))

    # for k in range(1,12):
    #     gibbs_sampling(args, model, val_dataloader, val_dataloader, eval_dataloader, early_stop_step=36, K=k)

    # for n_groups in [12]:
    #     for k in range(1,12):
    #         gibbs_sampling(
    #             args, model, val_dataloader, eval_dataloader, early_stop_step=36, K=k,
    #             n_groups=n_groups, annealing=True
    #         )
    
    # for k in [[14,8,2],[12,8,4],[10,8,6],[8,8,8],[6,8,10],[4,8,12],[2,8,14]]:
    #     score, sparisity, head_mask = gibbs_sampling_test(
    #         args, model, val_dataloader, eval_dataloader, early_stop_step=36, K=k, n_groups=3
    #     )

    # score, sparisity, head_mask = gibbs_sampling_test(
    #     args, model, val_dataloader, eval_dataloader, early_stop_step=36, K=[4,8,12], n_groups=3
    # )

    # score, sparisity, head_mask = gibbs_sampling_test(
    #     args, model, val_dataloader, eval_dataloader, early_stop_step=36, K=[8,8,8], n_groups=3
    # )

    # score, sparisity, head_mask = gibbs_sampling_test(
    #     args, model, val_dataloader, eval_dataloader, early_stop_step=36, K=[0,12], n_groups=2
    # )