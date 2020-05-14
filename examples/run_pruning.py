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
import matplotlib.pylab as plt
from sklearn.metrics import auc
from mlxtend.evaluate import permutation_test
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DefaultDataCollator,
    GlueDataset,
    glue_compute_metrics,
    glue_output_modes,
    glue_processors,
    set_seed,
)


logger = logging.getLogger(__name__)


def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def train(args, model, train_dataloader):
    if args.n_gpu > 1:
        n_layers, n_heads = model.module.config.num_hidden_layers, model.module.config.num_attention_heads
    else:
        n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_mask = torch.ones(n_layers, n_heads).to(args.device)
    head_mask.requires_grad_(requires_grad=True)

    optimizer = torch.optim.SGD([head_mask], lr=0.01, momentum=0.9)

    for epoch in range(3):
        running_loss = 0.0
        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
            for k, v in inputs.items():
                inputs[k] = v.to(args.device)
            
            loss = model(**inputs, head_mask=head_mask)[0]
            loss.backward()
            optimizer.step()
            head_mask.data = head_mask.data.clamp(min=0, max=1)

            running_loss += loss.item()
            if step % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 2000))
                running_loss = 0.0
    
    np.save(os.path.join(args.output_dir, "trained_head_mask.npy"), head_mask.detach().cpu().numpy())
    return head_mask

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
            preds = logits.detach().detach().cpu().numpy()
            labels = inputs["labels"].detach().detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs["labels"].detach().detach().cpu().numpy(), axis=0)
    
    preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    score = glue_compute_metrics(args.task_name, preds, labels)[args.metric_name]
        
    return score

def compute_heads_importance(
    args, model, train_dataloader, head_mask=None, zeros=False, save=True
):
    """ This method shows how to compute:
        - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    if args.n_gpu > 1:
        n_layers, n_heads = model.module.config.num_hidden_layers, model.module.config.num_attention_heads
    else:
        n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)

    if head_mask is None:
        if zeros:
            head_mask = torch.zeros(n_layers, n_heads).to(args.device)
        else:
            head_mask = torch.ones(n_layers, n_heads).to(args.device)
    head_mask.requires_grad_(requires_grad=True)
    tot_tokens = 0.0

    for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        if args.dont_use_abs:
            loss = model(**inputs, head_mask=head_mask)[0]
            first_derivate = torch.autograd.grad(loss, head_mask)[0]
            head_importance += first_derivate.detach()
        elif args.use_second:
            def func(head_mask):
                return model(**inputs, head_mask=head_mask)[0]
            H = torch.autograd.functional.hessian(func, head_mask).view((n_heads*n_layers, -1)).diagonal().view_as(head_mask)
            head_importance += H.detach()
            # first_derivate = torch.autograd.grad(loss, head_mask, create_graph=True)[0]
            # second_derivate = [torch.autograd.grad(f_d, head_mask, retain_graph=True)[0] for f_d in first_derivate.view(-1)]
            # second_derivate = torch.diag(torch.cat(second_derivate).view((n_layers*n_heads,-1))).view_as(first_derivate)
            # head_importance += second_derivate.detach().data
            # del first_derivate
        else:
            loss = model(**inputs, head_mask=head_mask)[0]
            first_derivate = torch.autograd.grad(loss, head_mask)[0]
            head_importance += first_derivate.abs().detach()

        tot_tokens += inputs["attention_mask"].float().detach().sum().data

    # Normalize
    head_importance /= tot_tokens
    # Layerwise importance normalization
    if not args.dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    if not args.dont_normalize_global_importance:
        head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())

    # Print/save matrices
    if save:
        file_name = "head_importance" + ("_second" if args.use_second else "")
        file_name += "_zeros" if zeros else ""
        np.save(os.path.join(args.output_dir, file_name + ".npy"), head_importance.detach().detach().cpu().numpy())

    # logger.info("Head importance scores")
    # print_2d_tensor(head_importance)
    # logger.info("Head ranked by importance scores")
    # head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=args.device)
    # head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
    #     head_importance.numel(), device=args.device
    # )
    # head_ranks = head_ranks.view_as(head_importance)
    # print_2d_tensor(head_ranks)

    return head_importance

def compute_S(
    args, model, train_dataloader, cross_layer=False
):
    """ This method shows how to compute:
        - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    tot_tokens = 0.0

    for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(**inputs)
        if args.use_contexts:
            all_weights = outputs[-2]
        else:
            all_weights = outputs[-1]

        if cross_layer:
            weights = all_weights
            if args.use_contexts:
                new_shape = weights[0].size()[:-1] + (12,-1) 
                weights = [w.detach().view(*new_shape) for w in weights] # (batch, token, head * vector_size) -> (batch, token, head, vector_size)
                weights = torch.cat(weights, 2) # (batch, token, layer * head, vector_size)
            else:
                weights = torch.cat(weights,1) # (batch, layer * head, token, vector_size)
                weights = weights.detach().transpose(1,2) # (batch, layer * head, token, vector_size) -> (batch, token, layer * head, vector_size)
            
            weights = torch.nn.functional.normalize(weights, dim=-1) # Normalize to 2_norm = 1
            weights = weights.reshape(-1, weights.shape[-2], weights.shape[-1]) # (batch * token, layer * head, vector_size)
            S = torch.bmm(weights, weights.transpose(1,2)) # (batch * token, layer * head, layer * head)
            S = S[inputs["attention_mask"].view(-1) == 1].sum(0) # Exclude masked tokens, (layer * head, layer * head)
            if step == 0:
                all_S = S
            else:
                all_S += S
        else:
            if step == 0:
                all_S = []
            for layer, weights in enumerate(all_weights):
                if args.use_contexts:
                    new_shape = weights.size()[:-1] + (12,-1) 
                    weights = weights.detach().view(*new_shape) # (batch, token, head * vector_size) -> (batch, token, head, vector_size)
                else:
                    weights = weights.detach().transpose(1,2) # (batch, head, token, vector_size) -> (batch, token, head, vector_size)
                weights = torch.nn.functional.normalize(weights, dim=-1) # Normalize to 2_norm = 1
                weights = weights.reshape(-1, weights.shape[-2], weights.shape[-1]) # (batch * token, head, vector_size)
                S = torch.bmm(weights, weights.transpose(1,2)) # (batch * token, head, head)
                S = S[inputs["attention_mask"].view(-1) == 1].sum(0) # Exclude masked tokens, (head, head)
                if step == 0:
                    all_S.append(S)
                else:
                    all_S[layer] += S

        tot_tokens += inputs["attention_mask"].float().detach().sum().data

    # Normalize
    if cross_layer:
        all_S = all_S / tot_tokens
    else:
        all_S = [S / tot_tokens for S in all_S]

    file_name = "all_S" + ("_contexts" if args.use_contexts else "")
    file_name = file_name + ("_cross_layer" if cross_layer else "")
    if cross_layer:
        np.save(os.path.join(args.output_dir, file_name + ".npy"), all_S.cpu().numpy())
    else:
        np.save(os.path.join(args.output_dir, file_name + ".npy"), all_S)

    return all_S

def save_results(args, scores, sparsities, all_head_masks, file_name):
    np.save(os.path.join(args.output_dir, file_name+'_scores.npy'), scores)
    np.save(os.path.join(args.output_dir, file_name+'_sparsities.npy'), sparsities)
    np.save(os.path.join(args.output_dir, file_name+'_masks.npy'), all_head_masks)

def mask_layers(
    args, model, eval_dataloader, file_name='masking_layers'
):
    if args.n_gpu > 1:
        n_layers, n_heads = model.module.config.num_hidden_layers, model.module.config.num_attention_heads
    else:
        n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    original_score = evaluate(args, model, eval_dataloader)
    logger.info("Pruning: original score: %f", original_score)

    new_head_mask = torch.ones(n_layers, n_heads).to(args.device)
    masked_layers = []
    scores = []
    scores.append(original_score)
    while new_head_mask.sum() != 0:
        max_score = float("-Inf")
        for i in range(n_layers):
            if i in masked_layers:
                continue
            head_mask = new_head_mask.clone()
            head_mask[i,:] = 0
            current_score = evaluate(args, model, eval_dataloader, head_mask=head_mask.clone())
            if current_score > max_score:
                max_score = current_score
                mask_layer = i
        masked_layers.append(mask_layer)
        scores.append(max_score)
        new_head_mask[mask_layer,:] = 0
        logger.info("Masking layer: %d, current score: %f", mask_layer, max_score)

    return scores

def mask_heads_random(
    args, model, eval_dataloader, file_name='masking_random', early_stop_step=None
):
    if args.n_gpu > 1:
        n_layers, n_heads = model.module.config.num_hidden_layers, model.module.config.num_attention_heads
    else:
        n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    original_score = evaluate(args, model, eval_dataloader)
    logger.info("Pruning: original score: %f", original_score)

    new_head_mask = torch.ones(n_layers, n_heads).to(args.device)
    num_to_mask = 12 # max(1, int(new_head_mask.numel() * args.masking_amount))

    scores = []
    sparsities = []
    all_head_masks = []

    current_score = original_score
    sparsities.append(0.0)
    scores.append(current_score * 100)
    all_head_masks.append(new_head_mask.data)

    step = 0

    all_heads = list(range(n_layers * n_heads))
    random.shuffle(all_heads)

    # while current_score >= original_score * args.masking_threshold:
    while new_head_mask.sum() != 0:
        if early_stop_step and step >= early_stop_step:
            break

        head_mask = new_head_mask.clone()  # save current head mask
        current_heads_to_mask = all_heads[step*num_to_mask:(step+1)*num_to_mask]
        # mask heads
        logger.info("Heads to mask: %s", str(current_heads_to_mask))
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_mask] = 0.0
        new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)

        current_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

        sparsity = 100 - new_head_mask.sum() / new_head_mask.numel() * 100
        scores.append(current_score * 100)
        sparsities.append(sparsity)
        all_head_masks.append(new_head_mask.data)

        logger.info(
            "Masking: current score: %f, remaning heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            100 - sparsity,
        )

        step += 1


    logger.info("Final head mask")
    print_2d_tensor(new_head_mask)
    save_results(args, scores, sparsities, all_head_masks, file_name)

    return scores, sparsities, all_head_masks

def mask_heads(
    args, model, train_dataloader, eval_dataloader, file_name='masking', early_stop_step=None,
    head_importance=None
):
    """ This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    if head_importance is None:
        head_importance = compute_heads_importance(args, model, train_dataloader)
    original_score = evaluate(args, model, eval_dataloader)
    logger.info("Pruning: original score: %f", original_score)

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = 12 # max(1, int(new_head_mask.numel() * args.masking_amount))

    scores = []
    sparsities = []
    all_head_masks = []

    current_score = original_score
    sparsities.append(0.0)
    scores.append(current_score * 100)
    all_head_masks.append(new_head_mask.data)

    step = 0

    # while current_score >= original_score * args.masking_threshold:
    while new_head_mask.sum() != 0:
        if early_stop_step and step >= early_stop_step:
            break

        head_mask = new_head_mask.clone()  # save current head mask
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            break

        # mask heads
        current_heads_to_mask = current_heads_to_mask[:num_to_mask]
        logger.info("Heads to mask: %s", str(current_heads_to_mask.tolist()))
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_mask] = 0.0
        new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        if args.exact_pruning and new_head_mask.sum() != 0:
            head_importance = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        current_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

        sparsity = 100 - new_head_mask.sum() / new_head_mask.numel() * 100
        scores.append(current_score * 100)
        sparsities.append(sparsity)
        all_head_masks.append(new_head_mask.data)

        logger.info(
            "Masking: current score: %f, remaning heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            100 - sparsity,
        )

        step += 1


    logger.info("Final head mask")
    print_2d_tensor(new_head_mask)

    file_name += "_contexts" if args.use_contexts else ""
    file_name += "_second" if args.use_second else ""
    file_name += "_no_abs" if args.dont_use_abs else ""
    save_results(args, scores, sparsities, all_head_masks, file_name)

    return scores, sparsities, all_head_masks

def unmask_heads(
    args, model, train_dataloader, eval_dataloader, file_name='unmasking', early_stop_step=None,
    head_importance=None
):
    if args.n_gpu > 1:
        n_layers, n_heads = model.module.config.num_hidden_layers, model.module.config.num_attention_heads
    else:
        n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    new_head_mask = torch.zeros(n_layers, n_heads).to(args.device)
    num_to_unmask = 12 # max(1, int(new_head_mask.numel() * args.masking_amount))

    if head_importance is None:
        if args.exact_pruning:
            head_importance = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        else:
            head_importance = compute_heads_importance(args, model, train_dataloader)

    current_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

    scores = []
    sparsities = []
    all_head_masks = []

    sparsities.append(100)
    scores.append(current_score * 100)
    all_head_masks.append(new_head_mask.data)

    step = 0

    while new_head_mask.sum() != new_head_mask.numel():
        if early_stop_step and step >= early_stop_step:
            break

        head_mask = new_head_mask.clone()  # save current head mask
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 1.0] = float("-Inf")
        current_heads_to_unmask = head_importance.view(-1).sort(descending = True)[1]

        if len(current_heads_to_unmask) <= num_to_unmask:
            break

        # unmask heads
        current_heads_to_unmask = current_heads_to_unmask[:num_to_unmask]
        logger.info("Heads to unmask: %s", str(current_heads_to_unmask.tolist()))
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_unmask] = 1.0
        new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        if args.exact_pruning and new_head_mask.sum() != new_head_mask.numel():
            head_importance = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone(), save=False
            )
        current_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

        sparsity = 100 - new_head_mask.sum() / new_head_mask.numel() * 100
        scores.append(current_score * 100)
        sparsities.append(sparsity)
        all_head_masks.append(new_head_mask.data)

        logger.info(
            "Masking: current score: %f, remaning heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            100 - sparsity,
        )

        step += 1
    
    logger.info("Final head mask")
    print_2d_tensor(new_head_mask)

    save_results(args, scores, sparsities, all_head_masks, file_name)

    return scores, sparsities, all_head_masks

def unmask_heads_per_layer(
    args, model, train_dataloader, eval_dataloader, file_name='unmasking_per_layer', early_stop_step=None,
    head_importance=None
):
    if args.n_gpu > 1:
        n_layers, n_heads = model.module.config.num_hidden_layers, model.module.config.num_attention_heads
    else:
        n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    new_head_mask = torch.zeros(n_layers, n_heads).to(args.device)

    if head_importance is None:
        if args.exact_pruning:
            head_importance = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        else:
            head_importance = compute_heads_importance(args, model, train_dataloader)
    current_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

    scores = []
    sparsities = []
    all_head_masks = []

    sparsities.append(100)
    scores.append(current_score * 100)
    all_head_masks.append(new_head_mask.data)

    step = 0

    while new_head_mask.sum() != new_head_mask.numel():
        if early_stop_step and step >= early_stop_step:
            break

        head_mask = new_head_mask.clone()  # save current head mask
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 1.0] = float("-Inf")
        current_heads_to_unmask = head_importance.sort(descending = True)[1][:,0]
        current_heads_to_unmask = current_heads_to_unmask + \
                                torch.arange(n_layers).to(args.device) * n_heads

        logger.info("Heads to unmask: %s", str(current_heads_to_unmask.tolist()))

        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_unmask] = 1.0
        new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        if args.exact_pruning and new_head_mask.sum() != new_head_mask.numel():
            head_importance = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        current_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

        sparsity = 100 - new_head_mask.sum() / new_head_mask.numel() * 100
        scores.append(current_score * 100)
        sparsities.append(sparsity)
        all_head_masks.append(new_head_mask.data)

        logger.info(
            "Masking: current score: %f, remaning heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            100 - sparsity,
        )

        step += 1
    
    logger.info("Final head mask")
    print_2d_tensor(new_head_mask)
    
    save_results(args, scores, sparsities, all_head_masks, file_name)

    return scores, sparsities, all_head_masks

def unmask_dpp_per_layer(
    args, model, train_dataloader, eval_dataloader, file_name='unmasking_dpp_per_layer', early_stop_step=None,
    head_importance=None, all_S=None
):
    if args.n_gpu > 1:
        n_layers, n_heads = model.module.config.num_hidden_layers, model.module.config.num_attention_heads
    else:
        n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    new_head_mask = torch.zeros(n_layers, n_heads).to(args.device)

    if head_importance is None:
        if args.exact_pruning:
            head_importance = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        else:
            head_importance = compute_heads_importance(args, model, train_dataloader)
    current_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

    if all_S is None:
        all_S = compute_S(args, model, train_dataloader)

    scores = []
    sparsities = []
    all_head_masks = []

    sparsities.append(100)
    scores.append(current_score * 100)
    all_head_masks.append(new_head_mask.data)

    step = 0

    while new_head_mask.sum() != new_head_mask.numel():
        if early_stop_step and step >= early_stop_step:
            break

        head_mask = new_head_mask.clone()  # save current head mask

        if step > 0:
            for layer, S in enumerate(all_S):
                activated_heads = head_mask[layer].nonzero().detach().cpu().numpy().reshape(-1)
                max_prob = float("-Inf")
                for i in range(n_heads):
                    if i in activated_heads:
                        continue
                    idx = np.sort(np.append(activated_heads, i))
                    entries = [[x, y] for x in idx for y in idx]
                    entries = list(zip(*entries))
                    S_subset = S[entries].view(len(idx), len(idx))
                    diversity = np.linalg.det(S_subset.cpu())
                    quality = torch.prod(head_importance[layer][idx]**2).cpu()
                    prob = diversity * quality
                    if prob > max_prob:
                        max_prob = prob
                        new_head = i
                new_head_mask[layer][new_head] = 1.0
                logger.info("Heads to unmask for layer %d: %d", layer, new_head)
        else:
            #head_importance[head_mask == 1.0] = float("-Inf")
            current_heads_to_unmask = head_importance.sort(descending = True)[1][:,0]
            current_heads_to_unmask = current_heads_to_unmask + \
                                    torch.arange(n_layers).to(args.device) * n_heads

            logger.info("Heads to unmask: %s", str(current_heads_to_unmask.tolist()))

            new_head_mask = new_head_mask.view(-1)
            new_head_mask[current_heads_to_unmask] = 1.0
            new_head_mask = new_head_mask.view_as(head_mask)
        
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        if args.exact_pruning and new_head_mask.sum() != new_head_mask.numel():
            head_importance = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        current_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

        sparsity = 100 - new_head_mask.sum() / new_head_mask.numel() * 100
        scores.append(current_score * 100)
        sparsities.append(sparsity)
        all_head_masks.append(new_head_mask.data)

        logger.info(
            "Masking: current score: %f, remaning heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            100 - sparsity,
        )

        step += 1
    
    logger.info("Final head mask")
    print_2d_tensor(new_head_mask)

    file_name += "_contexts" if args.use_contexts else ""
    file_name += "_second" if args.use_second else ""
    file_name += "_no_abs" if args.dont_use_abs else ""
    save_results(args, scores, sparsities, all_head_masks, file_name)

    return scores, sparsities, all_head_masks

def unmask_dpp(    
    args, model, train_dataloader, eval_dataloader, file_name='unmasking_dpp', early_stop_step=None,
    head_importance=None, all_S=None
):
    if args.n_gpu > 1:
        n_layers, n_heads = model.module.config.num_hidden_layers, model.module.config.num_attention_heads
    else:
        n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    new_head_mask = torch.zeros(n_layers, n_heads).to(args.device)
    num_to_unmask = 12

    if head_importance is None:
        if args.exact_pruning:
            head_importance = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        else:
            head_importance = compute_heads_importance(args, model, train_dataloader)
    current_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

    if all_S is None:
        all_S = compute_S(args, model, train_dataloader, cross_layer=True)

    scores = []
    sparsities = []
    all_head_masks = []

    sparsities.append(100)
    scores.append(current_score * 100)
    all_head_masks.append(new_head_mask.data)

    step = 0

    while new_head_mask.sum() != new_head_mask.numel():
        if early_stop_step and step >= early_stop_step:
            break

        head_mask = new_head_mask.clone()  # save current head mask

        new_head_mask = new_head_mask.view(-1)
        if step > 0:
            current_heads_to_unmask = []
            for j in range(num_to_unmask):
                activated_heads = new_head_mask.nonzero().cpu().numpy()
                max_prob = float("-Inf")
                for i in range(n_heads * n_layers):
                    if i in activated_heads:
                        continue
                    idx = np.sort(np.append(activated_heads, i))
                    entries = [[x, y] for x in idx for y in idx]
                    entries = list(zip(*entries))
                    S_subset = all_S[entries].view(len(idx), len(idx))
                    diversity = np.linalg.det(S_subset.cpu())
                    quality = torch.prod(head_importance.view(-1)[idx]**2).cpu()
                    prob = diversity * quality
                    if prob > max_prob:
                        max_prob = prob
                        new_head = i
                new_head_mask[new_head] = 1.0
                current_heads_to_unmask.append(new_head)
            logger.info("Heads to unmask: %s", str(current_heads_to_unmask))
        else:
            current_heads_to_unmask = head_importance.view(-1).sort(descending = True)[1]
            current_heads_to_unmask = current_heads_to_unmask[:num_to_unmask]
            logger.info("Heads to unmask: %s", str(current_heads_to_unmask.tolist()))
            new_head_mask[current_heads_to_unmask] = 1.0
        
        new_head_mask = new_head_mask.view_as(head_mask)
        
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        if args.exact_pruning and new_head_mask.sum() != new_head_mask.numel():
            head_importance = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        current_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

        sparsity = 100 - new_head_mask.sum() / new_head_mask.numel() * 100
        scores.append(current_score * 100)
        sparsities.append(sparsity)
        all_head_masks.append(new_head_mask.data)

        logger.info(
            "Masking: current score: %f, remaning heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            100 - sparsity,
        )

        step += 1
    
    logger.info("Final head mask")
    print_2d_tensor(new_head_mask)

    file_name += "_contexts" if args.use_contexts else ""
    file_name += "_second" if args.use_second else ""
    file_name += "_no_abs" if args.dont_use_abs else ""
    save_results(args, scores, sparsities, all_head_masks, file_name)

    return scores, sparsities, all_head_masks

def plot_graph(args, scores, sparsities, file_name, label=False):
    plt.plot(sparsities, scores, marker='o', label=file_name)

    if label:
        for x, y in zip(sparsities, scores):
            label = "{:.1f}".format(y)
            plt.annotate(label, (x, y), textcoords="offset points", ha='center', xytext=(0,10))
    
    plt.title("sparsity vs. accuracy")
    plt.xlabel("sparsity (%)")
    plt.ylabel("accuracy (%)")

    plt.xticks(np.arange(0,110,10))
    plt.yticks(np.arange(0,110,10))
    
    plt.grid(True)
    plt.legend()
    plt.show()

    png_file = os.path.join(args.output_dir, file_name+'.png')
    plt.savefig(png_file)

def test(args, model, train_dataloader, eval_dataset, K=10):
    lims = np.linspace(0, len(eval_dataset), K+1).astype('int')
    A = []
    B = []
    for i in range(K):
        sub_dataset = Subset(eval_dataset, list(range(lims[i], lims[i+1])))
        sub_sampler = SequentialSampler(sub_dataset) if args.local_rank == -1 else DistributedSampler(sub_dataset)
        sub_dataloader = DataLoader(
            sub_dataset, sampler=sub_sampler, batch_size=args.batch_size, collate_fn=DefaultDataCollator().collate_batch
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
    model = AutoModelForSequenceClassification.from_pretrained(
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

    # Prepare dataset for the GLUE task
    train_dataset = GlueDataset(args, tokenizer=tokenizer, local_rank=args.local_rank)
    if args.data_subset > 0:
        train_dataset = Subset(train_dataset, list(range(min(args.data_subset, len(train_dataset)))))
    train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=DefaultDataCollator().collate_batch
    )

    eval_dataset = GlueDataset(args, tokenizer=tokenizer, evaluate=True, local_rank=args.local_rank)
    if args.data_subset > 0:
        eval_dataset = Subset(eval_dataset, list(range(min(args.data_subset, len(eval_dataset)))))
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=DefaultDataCollator().collate_batch
    )

    # p_value = test(args, model, train_dataloader, eval_dataset)
    # logger.info("p_value is: %f", p_value)

    # Try head masking (set heads to zero until the score goes under a threshole)
    # and head pruning (remove masked heads and see the effect on the network)
    head_importance = compute_heads_importance(args, model, train_dataloader)
    # head_importance = torch.Tensor(np.load(os.path.join(args.output_dir, "head_importance.npy"))).to(args.device)
    # scores, sparsities, all_head_masks = mask_heads(
    #     args, model, train_dataloader, eval_dataloader, head_importance=head_importance.clone()
    # )
    # logger.info("Area under curve: %.2f", auc(sparsities, scores))
    
    # scores, sparsities, all_head_masks = unmask_heads(
    #     args, model, train_dataloader, eval_dataloader, head_importance=head_importance.clone()
    # )
    # logger.info("Area under curve: %.2f", auc(sparsities, scores))

    # scores, sparsities, all_head_masks = unmask_heads_per_layer(
    #     args, model, train_dataloader, eval_dataloader, head_importance=head_importance.clone()
    # )
    # logger.info("Area under curve: %.2f", auc(sparsities, scores))

    if False: #os.path.exists(os.path.join(args.output_dir, "all_S.npy")):
        all_S = np.load(os.path.join(args.output_dir, "all_S.npy"), allow_pickle=True)
    else:
        all_S = compute_S(args, model, train_dataloader)
    scores, sparsities, all_head_masks = unmask_dpp(
        args, model, train_dataloader, eval_dataloader, head_importance=head_importance.clone(), all_S=all_S
    )
    logger.info("Area under curve: %.2f", auc(sparsities, scores))

    args.use_contexts = True
    if False: #os.path.exists(os.path.join(args.output_dir, "all_S_contexts.npy")):
        all_S_contexts = np.load(os.path.join(args.output_dir, "all_S_contexts.npy"), allow_pickle=True)
    else:
        all_S_contexts = compute_S(args, model, train_dataloader)
    scores, sparsities, all_head_masks = unmask_dpp(
        args, model, train_dataloader, eval_dataloader, head_importance=head_importance.clone(), all_S=all_S_contexts
    )
    logger.info("Area under curve: %.2f", auc(sparsities, scores))

    # args.use_second = True
    # head_importance = compute_heads_importance(args, model, train_dataloader)
    # scores, sparsities, all_head_masks = unmask_dpp(
    #     args, model, train_dataloader, eval_dataloader, head_importance=head_importance.clone(), all_S=all_S_contexts
    # )
    # logger.info("Area under curve: %.2f", auc(sparsities, scores))

    # args.use_contexts = False
    # scores, sparsities, all_head_masks = unmask_dpp(
    #     args, model, train_dataloader, eval_dataloader, head_importance=head_importance.clone(), all_S=all_S
    # )
    # logger.info("Area under curve: %.2f", auc(sparsities, scores))

    # head_mask = train(args, model, train_dataloader)
    # result = evaluate(args, model, eval_dataloader, head_mask=head_mask.clone())
    # logger.info(result)

if __name__ == "__main__":
    main()
