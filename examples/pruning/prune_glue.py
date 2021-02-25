#!/usr/bin/env python3
# Copyright 2018 CMU and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Bertology: this script shows how you can explore the internals of the models in the library to:
    - compute the importance of each head
    - prune (remove) the low importance head.
    Some parts of this script are adapted from the code of Michel et al. (http://arxiv.org/abs/1905.10650)
    which is available at https://github.com/pmichel31415/are-16-heads-really-better-than-1
"""
import argparse
import logging
import os
from datetime import datetime
import re

import matplotlib.pylab as plt
from sklearn.metrics import auc
from mlxtend.evaluate import permutation_test
import random
import sampling_utils

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from pruning_utils import *

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    GlueDataset,
    glue_compute_metrics,
    glue_output_modes,
    glue_processors,
    set_seed,
    AdamW,
    BertForSequenceClassificationConcrete,
)


logger = logging.getLogger(__name__)

def test(args, model, train_dataloader, eval_dataset, K=10):
    lims = np.linspace(0, len(eval_dataset), K+1).astype('int')
    A = []
    B = []
    for i in range(K):
        sub_dataset = Subset(eval_dataset, list(range(lims[i], lims[i+1])))
        sub_sampler = SequentialSampler(sub_dataset) if args.local_rank == -1 else DistributedSampler(sub_dataset)
        sub_dataloader = DataLoader(
            sub_dataset, sampler=sub_sampler, batch_size=args.batch_size, collate_fn=default_data_collator
        )
        if i == 0:
            A_scores, A_sparsities, A_head_masks = mask_heads(args, model, train_dataloader, sub_dataloader)
            A.append(auc(A_sparsities, A_scores).cpu())
            B_scores, B_sparsities, B_head_masks = unmask_dpp(args, model, train_dataloader, sub_dataloader)
            B.append(auc(B_sparsities, B_scores).cpu())
        else:
            A_scores = []
            for head_mask in A_head_masks:
                A_scores.append(evaluate(args, model, sub_dataloader, head_mask=head_mask))
            A.append(auc(A_sparsities, A_scores).cpu())

            B_scores = []
            for head_mask in B_head_masks:
                B_scores.append(evaluate(args, model, sub_dataloader, head_mask=head_mask))
            B.append(auc(B_sparsities, B_scores).cpu())
    p_value = permutation_test(A, B,
                           method='exact', func=lambda x, y: np.mean(y) - np.mean(x))
    np.save(os.path.join(args.output_dir, "A.npy"), A)
    np.save(os.path.join(args.output_dir, "B.npy"), B)
    
    return p_value


def main():
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
        help="The name of the task to train selected in the list: " + ", ".join(glue_processors.keys()),
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
        "--use_squared", action="store_true", help="Use squared derivative as quality"
    )
    parser.add_argument(
        "--use_second", action="store_true", help="Use second order derivative as quality"
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

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in glue_processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = glue_processors[args.task_name]()
    args.output_mode = glue_output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        output_attentions=True,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, cache_dir=args.cache_dir,
    )
    model = BertForSequenceClassificationConcrete.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )
    model.to(args.device)
    model.eval()
    # Print/save training arguments
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(args, os.path.join(args.output_dir, "run_args.bin"))
    logger.info("Training/evaluation parameters %s", args)

    # Prepare dataset for the GLUE task
    train_dataset = GlueDataset(args, tokenizer=tokenizer)
    train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=default_data_collator
    )

    if args.task_name == "mnli":
        args.task_name="mnli-mm"
        eval_dataset = GlueDataset(args, tokenizer=tokenizer, mode="dev")
        args.task_name = "mnli"
    else:
        eval_dataset = GlueDataset(args, tokenizer=tokenizer, mode="dev")
    if args.data_subset > 0:
        eval_dataset = Subset(eval_dataset, list(range(min(args.data_subset, len(eval_dataset)))))
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=default_data_collator
    )

    # p_value = test(args, model, train_dataloader, eval_dataset)
    # logger.info("p_value is: %f", p_value)

    # Try head masking (set heads to zero until the score goes under a threshole)
    # and head pruning (remove masked heads and see the effect on the network)
    # head_importance = compute_heads_importance(args, model, train_dataloader)
    # head_importance = torch.Tensor(np.load(os.path.join(args.output_dir, "head_importance.npy"))).to(args.device)
    args.exact_pruning = True
    # args.dont_normalize_importance_by_layer = True
    # args.use_second = True
    scores, sparsities, all_head_masks = mask_heads(
        args, model, train_dataloader, eval_dataloader
    )
    logger.info("Area under curve: %.2f", auc(sparsities, scores))
    
    # scores, sparsities, all_head_masks = unmask_heads(
    #     args, model, train_dataloader, eval_dataloader
    # )
    # logger.info("Area under curve: %.2f", auc(sparsities, scores))

    # score, sparisity, head_mask = gibbs_sampling(
    #     args, model, train_dataloader, eval_dataloader, val_dataloader=val_dataloader, early_stop_step=24, K=2, n_groups=1
    # )

    # scores = []
    # sparsities = []
    # all_head_masks = []
    # for k in range(1,12):
    #     score, sparisity, head_mask = gibbs_sampling(
    #         args, model, val_dataloader, eval_dataloader, early_stop_step=36, K=k, n_groups=1
    #     )
    # for k in range(1,12):
    for head_mask in all_head_masks:
        model = BertForSequenceClassificationConcrete.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
        # model.eval()

        # Distributed and parallel training
        # model.to(args.device)
        # if args.local_rank != -1:
        #     model = torch.nn.parallel.DistributedDataParallel(
        #         model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        #     )
        # elif args.n_gpu > 1:
        #     model = torch.nn.DataParallel(model)
        # score, sparisity, head_mask = gibbs_sampling(
        #     args, model, val_dataloader, eval_dataloader, early_stop_step=36, K=k, n_groups=1, annealing=False
        # )
        # score, sparsity, head_mask = random_sampling(
        #     args, model, eval_dataloader, val_dataloader, early_stop_step=36, K=k, n_groups=1
        # )
        score = train(args, model, head_mask, train_dataset, eval_dataset, epoch=3.0)
        logger.info(
            "Current score: %f, remaining heads %d",
            score,
            head_mask.sum(),
        )
        # scores.append(score)
        # sparsities.append(sparisity)
        # all_head_masks.append(head_mask)
    # save_results(args, scores, sparsities, all_head_masks, "markov_small")

    # scores = []
    # sparsities = []
    # all_head_masks = []
    # for n_groups in [6]:
    #     for k in range(8,12):
    #         score, sparisity, head_mask = gibbs_sampling(
    #             args, model, val_dataloader, eval_dataloader, early_stop_step=36, K=k, n_groups=n_groups,
    #             annealing=True
    #         )

    # for n_groups in [1,2,3,4,6,12]:
    #     for k in [11]:
    #         score, sparisity, head_mask = gibbs_sampling(
    #             args, model, val_dataloader, eval_dataloader, early_stop_step=36, K=k, n_groups=n_groups,
    #             annealing=True
    #         )
        # scores.append(score)
    #     sparsities.append(sparisity)
    #     all_head_masks.append(head_mask)
    # save_results(args, scores, sparsities, all_head_masks, "gibbs_layer_small")
    
    # for seed in [0,1,2,3]:
        # scores = []
        # sparsities = []
        # all_head_masks = []
        # for k in range(1,12):
        #     score, sparisity, head_mask = gibbs_sampling(
        #         args, model, val_dataloader, eval_dataloader, early_stop_step=36, K=k,
        #         n_groups=1, annealing=True, seed=seed
        #     )
            # scores.append(score)
            # sparsities.append(sparisity)
            # all_head_masks.append(head_mask)
        # save_results(args, scores, sparsities, all_head_masks, "gibbs_group"+str(n_groups)+"_seed"+str(seed))

    # for k in [2,3,4,5,6,7]:
    #     train(args, model, train_dataloader, eval_dataloader, K=k)


    # score, sparisity, head_mask = gibbs_sampling_test(
    #     args, model, val_dataloader, eval_dataloader, early_stop_step=36, K=[8,4], n_groups=2
    # )
    # plot_graph(args)

#     head_mask = "1.00000 0.00000 0.00000 1.00000 1.00000 0.00000 0.00000 0.00000 1.00000 0.00000 0.00000 0.00000 \
# 0.00000 0.00000 0.00000 0.00000 1.00000 0.00000 0.00000 1.00000 1.00000 1.00000 1.00000 1.00000 \
# 1.00000 1.00000 1.00000 1.00000 0.00000 0.00000 1.00000 0.00000 0.00000 1.00000 1.00000 1.00000 \
# 1.00000 0.00000 0.00000 1.00000 0.00000 1.00000 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 \
# 1.00000 1.00000 1.00000 1.00000 1.00000 1.00000 1.00000 0.00000 0.00000 1.00000 0.00000 0.00000 \
# 1.00000 0.00000 0.00000 1.00000 1.00000 1.00000 0.00000 1.00000 1.00000 0.00000 1.00000 0.00000 \
# 1.00000 1.00000 0.00000 0.00000 0.00000 1.00000 1.00000 0.00000 1.00000 0.00000 1.00000 0.00000 \
# 1.00000 1.00000 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 1.00000 0.00000 1.00000 0.00000 \
# 0.00000 1.00000 0.00000 0.00000 1.00000 1.00000 0.00000 1.00000 1.00000 1.00000 1.00000 1.00000 \
# 1.00000 0.00000 1.00000 0.00000 0.00000 0.00000 1.00000 1.00000 1.00000 0.00000 1.00000 1.00000 \
# 1.00000 0.00000 0.00000 1.00000 0.00000 0.00000 0.00000 1.00000 1.00000 0.00000 1.00000 1.00000 \
# 1.00000 1.00000 1.00000 0.00000 0.00000 0.00000 0.00000 0.00000 1.00000 1.00000 0.00000 0.00000"
#     head_mask = np.array(re.split(" ", head_mask)).reshape(12,12).astype("float")
#     head_mask = torch.Tensor(head_mask).to(args.device)
#     logger.info(evaluate(args, model, eval_dataloader, head_mask))
    # random_sampling(args, model, eval_dataloader, val_dataloader, early_stop_step=36, K=6, n_groups=1)
if __name__ == "__main__":
    main()
