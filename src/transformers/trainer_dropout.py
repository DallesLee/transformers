import inspect
import json
import math
import os
import re
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange

from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .file_utils import is_datasets_available, is_torch_tpu_available
from .integrations import (
    default_hp_search_backend,
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
)
from .modeling_utils import PreTrainedModel
from .optimization import AdamW, get_linear_schedule_with_warmup
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    distributed_broadcast_scalars,
    distributed_concat,
    set_seed,
)
from .training_args import TrainingArguments
from .utils import logging
from .trainer import (
    torch_distributed_zero_first,
    SequentialDistributedSampler,
    get_tpu_sampler,
    Trainer
)

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboardX import SummaryWriter

if is_wandb_available():
    import wandb

if is_comet_available():
    import comet_ml

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)

class DropoutTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        # num_of_heads: Optional[int] = 36,
        # temperature: Optional[float] = 1.0,
        # w_lr: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics,
            tb_writer, optimizers, **kwargs
        )
        # self.num_of_heads = num_of_heads
        # self.temperature = temperature
        # self.w_lr = w_lr
        # self.w = nn.Parameter(torch.empty([12,12]).to(model.device))
        # nn.init.xavier_uniform_(self.w)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        # if self.w_lr is None:
        #     self.optimizer.add_param_group({"params": self.w})
        # else:
        #     self.optimizer.add_param_group({
        #         "params": self.w,
        #         "lr": self.w_lr,
        #     })
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )

    # def gumbel_soft_top_k(self, w, k, t):
    #     # apply gumbel noise
    #     u = torch.rand_like(w)
    #     r = -torch.log(-torch.log(u)) + w
    #     epsilon = torch.ones_like(r)
    #     epsilon *= np.finfo(np.float32).tiny

    #     # soft top k
    #     p = torch.zeros([k, w.size()[0]]).to(w.device)
    #     p[0] = torch.softmax(r/t,0)
    #     for j in range(1,k):
    #         r += torch.log(torch.max(1-p[j-1], epsilon))
    #         p[j] = torch.softmax(r / t, 0)
            
    #     return p.sum(0)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        if hasattr(self, "_training_step"):
            warnings.warn(
                "The `_training_step` method is deprecated and won't be called in a future version, define `training_step` in your subclass.",
                FutureWarning,
            )
            return self._training_step(model, inputs, self.optimizer)
        
        # if model._apply_gates:
        #     gates = torch.stack(model.get_gate_values())
        #     head_mask = self.gumbel_soft_top_k(gates.view(-1), self.num_of_heads, self.temperature).view_as(gates)
        # else:
        #     head_mask = self.gumbel_soft_top_k(self.w.view(-1), self.num_of_heads, self.temperature).view_as(self.w)
        # model.apply_masks(head_mask)

        with torch.autograd.set_detect_anomaly(True):
            model.train()
            inputs = self._prepare_inputs(inputs)

            if self.args.fp16 and _use_native_amp:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.args.fp16 and _use_native_amp:
                self.scaler.scale(loss).backward()
            elif self.args.fp16 and _use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

        return loss.detach()