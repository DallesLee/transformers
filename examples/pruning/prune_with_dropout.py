# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    DropoutTrainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
    BertForSequenceClassificationConcrete,
    BertForSequenceClassification,
    AdamW,
    default_data_collator
)
from torch.utils.data import DataLoader, SequentialSampler, Subset
from tqdm import tqdm
from pruning_utils import print_2d_tensor


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    def convert_gate_to_mask(gates, num_of_heads=None):
        if num_of_heads is not None:
            head_mask = torch.zeros_like(gates)
            current_heads_to_keep = gates.view(-1).sort(descending = True)[1]
            current_heads_to_keep = current_heads_to_keep[:num_of_heads]
            head_mask = head_mask.view(-1)
            head_mask[current_heads_to_keep] = 1.0
            head_mask = head_mask.view_as(gates)
        else:
            head_mask = (gates > 0.5).float()
        return head_mask

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )

    if data_args.task_name == "mnli":
        metric = "eval_mnli/acc"
    else:
        metric = "eval_acc"

    annealing = True
    reducing_heads = False
    for temperature in [1e-08]:
        for num_of_heads in [12]:
            for cooldown_steps in [10000]:
                for starting_temperature in [1.0]:
                    for starting_num_of_heads in [60]:
                        for lr in [2e-1]:
                            logger.info(
                                "cooldown_steps: {}, starting_temperature: {}, starting_num_of_heads: {}, learning_rate: {}".format(
                                    cooldown_steps if annealing or reducing_heads else "N.A.", 
                                    starting_temperature if annealing else "N.A.", 
                                    starting_num_of_heads if reducing_heads else "N.A.",
                                    lr,
                            ))
                            torch.manual_seed(42)
                            model = BertForSequenceClassificationConcrete.from_pretrained(
                                model_args.model_name_or_path,
                                config=config,
                            )

                            model.apply_dropout(num_of_heads, temperature)

                            optimizer_grouped_parameters = [
                                {
                                    "params": [p for n, p in model.named_parameters() if n != "w"],
                                    "lr": training_args.learning_rate,
                                },
                                {
                                    "params": [p for n, p in model.named_parameters() if n == "w"],
                                    "lr": lr,
                                },
                            ]
                            optimizer = AdamW(
                                optimizer_grouped_parameters,
                                betas=(0.9, 0.999),
                                eps=1e-8,
                            )

                            # Initialize our Trainer
                            training_args.max_steps = -1
                            trainer = DropoutTrainer(
                                model=model,
                                args=training_args,
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                compute_metrics=build_compute_metrics_fn(data_args.task_name),
                                num_of_heads=num_of_heads,
                                reducing_heads=reducing_heads,
                                temperature=temperature,
                                cooldown_steps=cooldown_steps,
                                annealing=annealing,
                                starting_temperature=starting_temperature,
                                starting_num_of_heads=starting_num_of_heads,
                                optimizers=(optimizer, None),
                                double=True,
                            )

                            # Training
                            trainer.train()
                            trainer.save_model()
                            score = trainer.evaluate(eval_dataset=eval_dataset)[metric]
                            print_2d_tensor(model.get_w())
                            logger.info("temperature: {}, num of heads: {}, accuracy: {}".format(temperature, num_of_heads, score * 100))

                            model._apply_dropout = False
                            list_of_nums = [12, 24, 36]
                            if num_of_heads not in list_of_nums:
                                list_of_nums.insert(0, num_of_heads)
                            for num_to_unmask in sorted(list_of_nums):
                                head_mask = convert_gate_to_mask(model.get_w(), num_to_unmask)
                                # print_2d_tensor(head_mask)
                                model.apply_masks(head_mask)
                                score = trainer.evaluate(eval_dataset=eval_dataset)[metric]
                                sparsity = 100 - head_mask.sum() / head_mask.numel() * 100
                                logger.info(
                                    "Masking: current score: %f, remaining heads %d (%.1f percents)",
                                    score,
                                    head_mask.sum(),
                                    100 - sparsity,
                                )
    


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
