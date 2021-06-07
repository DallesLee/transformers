import logging
import os

import matplotlib.pylab as plt
from sklearn.metrics import auc
import random
import sampling_utils

import numpy as np
import torch
from tqdm import tqdm

from transformers import glue_compute_metrics, EvalPrediction
from typing import Dict, List, Tuple
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler

from typing import Callable, Dict, Optional

from transformers import (
    glue_compute_metrics,
    glue_output_modes,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator
)

logger = logging.getLogger(__name__)

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, label_map) -> Tuple[List[int], List[int]]:
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list

def compute_metrics(p: EvalPrediction, label_map) -> Dict:
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids, label_map)
    return {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    output_mode = glue_output_modes[task_name]
    def compute_metrics_fn(p: EvalPrediction):
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(task_name, preds, p.label_ids)

    return compute_metrics_fn

def train(args, model, head_mask, train_dataset, eval_dataset, epoch=1.0):
    training_parser = HfArgumentParser((TrainingArguments))
    str_args = "--output_dir {} --learning_rate 2e-5 --per_device_train_batch_size 32\
     --do_train --do_eval --num_train_epochs {}".format(args.output_dir + "/" + str(head_mask.sum().cpu().numpy()), epoch)
    training_args = training_parser.parse_args_into_dataclasses(str_args.split())[0]
    model.train()
    model.apply_masks(head_mask)
    model.config.output_attentions = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(eval_dataset.args.task_name),
    )
    trainer.train()
    score = trainer.evaluate(eval_dataset=eval_dataset)["eval_" + args.metric_name]
    # logger.info("Accuracy: {}".format(score))
    return score

def dropout_train(args, model, train_dataset, eval_dataset, num_to_mask=36, epoch=3):
    training_parser = HfArgumentParser((TrainingArguments))
    str_args = "--output_dir output/ --learning_rate 2e-5 --per_device_train_batch_size 32\
     --do_train --do_eval --num_train_epochs 1.0"
    training_args = training_parser.parse_args_into_dataclasses(str_args.split())[0]

    train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, collate_fn=default_data_collator
    )

    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_mask = torch.ones(n_layers, n_heads).to(args.device)

    scores = []

    for i in range(epoch):
        model.eval()
        head_importance, _ = compute_heads_importance(args, model, train_dataloader, head_mask)
        current_heads_to_mask = head_importance.view(-1).sort()[1]
        current_heads_to_mask = current_heads_to_mask[:num_to_mask]
        logger.info("Heads to mask: %s", str(current_heads_to_mask.tolist()))
        head_mask = torch.ones(n_layers, n_heads).to(args.device)
        head_mask = head_mask.view(-1)
        head_mask[current_heads_to_mask] = 0.0
        head_mask = head_mask.view_as(head_importance)

        training_args.max_steps = -1
        model.train()
        model.apply_masks(head_mask)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(eval_dataset.args.task_name),
        )
        trainer.train()
        scores.append(trainer.evaluate(eval_dataset=eval_dataset)['eval_acc'])
    
    return scores

def evaluate(args, model, eval_dataloader, head_mask=None):
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(args.device)

    model.apply_masks(head_mask)
    # Evaluate
    preds = None
    labels = None
    for step, inputs in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(**inputs)
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
    
    if args.task_name in ['ner', 'pos']:
        score = compute_metrics(EvalPrediction(predictions=preds, label_ids=labels), model.config.id2label)['f1']
    else:
        preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
        score = glue_compute_metrics(args.task_name, preds, labels)[args.metric_name]
        
    return score

def compute_heads_importance(
    args, model, train_dataloader, head_mask=None, zeros=False
):
    """ This method shows how to compute:
        - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)

    if head_mask is None:
        if zeros:
            head_mask = torch.zeros(n_layers, n_heads).to(args.device)
        else:
            head_mask = torch.ones(n_layers, n_heads).to(args.device)
    head_mask.requires_grad_(requires_grad=True)
    model.apply_masks(head_mask)
    
    tot_tokens = 0.0
    preds = None
    labels = None
    for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(**inputs)
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last
        
        if args.dont_use_abs:
            first_derivative = torch.autograd.grad(loss, head_mask)[0]
            head_importance += first_derivative.detach()
        elif args.use_squared:
            first_derivative = torch.autograd.grad(loss, head_mask)[0]
            head_importance += first_derivative.pow(2).detach()
        elif args.use_second:
            first_derivative = torch.autograd.grad(loss, head_mask, create_graph=True)[0]
            second_derivative = torch.zeros_like(first_derivative)
            for _ in range(10):
                v = torch.rand(12,12).to(args.device)
                v[v>=0.5] = 1
                v[v<0.5] = -1
                second_derivative += (v * torch.autograd.grad((first_derivative * v).sum(), head_mask, retain_graph=True)[0]).detach()
            second_derivative /= 10
            head_importance -= second_derivative.data
        else:
            first_derivative = torch.autograd.grad(loss, head_mask)[0]
            head_importance += first_derivative.abs().detach()

        tot_tokens += inputs["attention_mask"].float().detach().sum().data

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().detach().cpu().numpy()
            labels = inputs["labels"].detach().detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs["labels"].detach().detach().cpu().numpy(), axis=0)

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
    file_name = "head_importance" + ("_second" if args.use_second else "")
    file_name += "_zeros" if zeros else ""
    np.save(os.path.join(args.output_dir, file_name + ".npy"), head_importance.detach().cpu().numpy())
    logger.info(args.task_name)
    if args.task_name in ['ner', 'pos']:
        score = compute_metrics(EvalPrediction(predictions=preds, label_ids=labels), model.config.id2label)['f1']
    else:
        preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
        score = glue_compute_metrics(args.task_name, preds, labels)[args.metric_name]

    return head_importance, score

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

    if not args.exact_pruning:
        file_name = "all_S" + ("_contexts" if args.use_contexts else "")
        file_name = file_name + ("_cross_layer" if cross_layer else "")
        if cross_layer:
            np.save(os.path.join(args.output_dir, file_name + ".npy"), all_S.cpu().numpy())
        else:
            np.save(os.path.join(args.output_dir, file_name + ".npy"), all_S)

    return all_S

def mask_layers(
    args, model, eval_dataloader, file_name='masking_layers'
):
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
        head_importance, _ = compute_heads_importance(args, model, train_dataloader)
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
    all_head_masks.append(new_head_mask.clone())

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
            head_importance, _ = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        current_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

        sparsity = 100 - new_head_mask.sum() / new_head_mask.numel() * 100
        scores.append(current_score * 100)
        sparsities.append(sparsity)
        all_head_masks.append(new_head_mask.clone())

        logger.info(
            "Masking: current score: %f, remaning heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            100 - sparsity,
        )

        if new_head_mask.sum() <= 12:
            num_to_mask = 1

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
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    new_head_mask = torch.zeros(n_layers, n_heads).to(args.device)
    num_to_unmask = 12 # max(1, int(new_head_mask.numel() * args.masking_amount))

    if head_importance is None:
        if args.exact_pruning:
            head_importance, _ = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        else:
            head_importance, _ = compute_heads_importance(args, model, train_dataloader)

    current_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

    scores = []
    sparsities = []
    all_head_masks = []

    sparsities.append(100)
    scores.append(current_score * 100)
    all_head_masks.append(new_head_mask.clone())

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
            head_importance, _ = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        current_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

        sparsity = 100 - new_head_mask.sum() / new_head_mask.numel() * 100
        scores.append(current_score * 100)
        sparsities.append(sparsity)
        all_head_masks.append(new_head_mask.clone())

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

def unmask_heads_f5(
    args, model, train_dataloader, eval_dataloader, file_name='unmasking_f5', early_stop_step=None,
    head_importance=None
):
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    new_head_mask = torch.zeros(n_layers, n_heads).to(args.device)
    num_to_unmask = 12 # max(1, int(new_head_mask.numel() * args.masking_amount))

    if head_importance is None:
        if args.exact_pruning:
            head_importance, _ = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        else:
            head_importance, _ = compute_heads_importance(args, model, train_dataloader)

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
        # head_importance[head_mask == 1.0] = float("-Inf")
        current_heads_to_unmask = head_importance.view(-1).sort(descending = True)[1]

        if len(current_heads_to_unmask) <= num_to_unmask:
            break

        # unmask heads
        current_heads_to_unmask = current_heads_to_unmask[:step * 12]
        logger.info("Heads to unmask: %s", str(current_heads_to_unmask.tolist()))
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[:] = 0.0
        new_head_mask[current_heads_to_unmask] = 1.0
        new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        if args.exact_pruning and new_head_mask.sum() != new_head_mask.numel():
            head_importance, _ = compute_heads_importance(
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

def unmask_heads_per_layer(
    args, model, train_dataloader, eval_dataloader, file_name='unmasking_per_layer', early_stop_step=None,
    head_importance=None
):
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    new_head_mask = torch.zeros(n_layers, n_heads).to(args.device)

    if head_importance is None:
        if args.exact_pruning:
            head_importance, _ = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        else:
            head_importance, _ = compute_heads_importance(args, model, train_dataloader)
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
            head_importance, _ = compute_heads_importance(
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
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    new_head_mask = torch.zeros(n_layers, n_heads).to(args.device)

    if head_importance is None:
        if args.exact_pruning:
            head_importance, _ = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        else:
            head_importance, _ = compute_heads_importance(args, model, train_dataloader)
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
            head_importance, _ = compute_heads_importance(
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
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    new_head_mask = torch.zeros(n_layers, n_heads).to(args.device)
    num_to_unmask = 12

    if head_importance is None:
        if args.exact_pruning:
            head_importance, _ = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        else:
            head_importance, _ = compute_heads_importance(args, model, train_dataloader)
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
            head_importance, _ = compute_heads_importance(
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

# Sampling
def log_elem_polynomials(log_lambdas, k):
    N = len(log_lambdas)
    E = np.full((k+1,N+1), sampling_utils.NEG_INF)
    E[0,:] = 0.                     # initialization
    for i in range(1, k+1):
        for n in range(1,N+1):
            interm = log_lambdas[n-1] + E[i-1,n-1]
            E[i,n] = sampling_utils.log_add(E[i,n-1], interm) 
    return E

def log_sample_k_dpp(log_lambdas, k, seed=0):
    N = len(log_lambdas)
    if k >= N:
        return range(N)
    np.random.seed(seed=seed)
    J = []
    log_E = log_elem_polynomials(log_lambdas, k)
    
    for n in range(N,0,-1):
        u = np.random.uniform()
        thresh = log_lambdas[n-1] + log_E[k-1,n-1] - log_E[k,n]        
        if np.log(u) < thresh:
            J.append(n-1)
            k -= 1
            if k == 0:
                break
    return J

def gibbs_sampling(    
    args, model, train_dataloader, eval_dataloader, val_dataloader=None,
    early_stop_step=2, K=4, annealing=False, T=1, n_groups=4, seed=0
):
    torch.manual_seed(seed)
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    layers_per_group = n_layers // n_groups

    num_to_unmask = K

    new_head_mask = torch.rand(n_layers, n_heads).to(args.device)
    new_head_mask[new_head_mask>=0.5] = 1.0
    new_head_mask[new_head_mask<0.5] = 0.0
    
    val_scores = []
    eval_scores = []
    sparsities = []
    all_head_masks = []

    step = 0
    T = 1

    while step <= early_stop_step:
        # Store results and prepare for next run
        if val_dataloader is None:
            head_importance, val_score = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        else:
            head_importance, _ = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
            val_score = evaluate(args, model, val_dataloader, head_mask=new_head_mask.clone())
        eval_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

        sparsity = 100 - new_head_mask.sum() / new_head_mask.numel() * 100
        logger.info(
            "Step %d: validation score: %f, evaluation score: %f, remaning heads %d (%.1f percents)",
            step,
            val_score,
            eval_score,
            new_head_mask.sum(),
            100 - sparsity,
        )

        val_scores.append(val_score * 100)
        eval_scores.append(eval_score * 100)
        sparsities.append(torch.round(sparsity))
        all_head_masks.append(new_head_mask.data)

        # Start pruning
        head_mask = new_head_mask.clone()  # save current head mask

        # if step < n_groups:
        #     group = step
        # else:
        #     group = random.randint(0,n_groups-1)
        group = step % n_groups

        indices = list(range(group*layers_per_group, (group+1)*layers_per_group))
        if not annealing:
            current_heads_to_unmask = head_importance[indices].view(-1).sort(descending = True)[1]
            current_heads_to_unmask = current_heads_to_unmask[:num_to_unmask] 
            current_heads_to_unmask += group * layers_per_group * n_heads
            logger.info("Heads to unmask for group %d: %s", group, str(current_heads_to_unmask.tolist()))
        else:
            lambdas = np.float128(head_importance[indices].view(-1).double().cpu().numpy())
            log_lambdas = np.log(lambdas) * (1/T)
            current_heads_to_unmask = log_sample_k_dpp(log_lambdas, num_to_unmask, seed)
            current_heads_to_unmask = [h + group * layers_per_group * n_heads for h in current_heads_to_unmask]
            logger.info("Heads to unmask for group %d: %s", group, str(current_heads_to_unmask))
        new_head_mask[indices] = 0.0
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_unmask] = 1.0
        new_head_mask = new_head_mask.view_as(head_mask)

        step += 1

        if T <= 1e-5:
            multiplicant = 1
        elif T <= 1000:
            multiplicant = 0.1

        if step % 4 == 0:
            T *= multiplicant  

    # file_name = "gibbs_per_group" + "_deterministic" if deterministic else ""
    # save_results(args, val_scores, sparsities, all_head_masks, file_name)

    val_scores = np.array(val_scores)
    desired_sparsity = np.round(100 - K / (n_heads * n_layers) * 100)
    best_head_mask = None
    best_score = None
    i = None
    while True:
        if i is not None and i == np.argmax(val_scores):
            logger.info("Not converged to desired sparsity yet")
            break
        else:
            i = np.argmax(val_scores)
        if sparsities[i] == desired_sparsity:
            best_score = eval_scores[i]
            best_head_mask = all_head_masks[i]
            logger.info("Best score from iteration %d: %f", i, best_score)
            logger.info("Best head mask")
            print_2d_tensor(best_head_mask)
            # np.save(os.path.join(args.output_dir, str(K) + '_best_mask_per_layer.npy'), best_head_mask.cpu())
            break
        else:
            val_scores[i] = float("-inf")

    return best_score, desired_sparsity, best_head_mask

def random_sampling(    
    args, model, eval_dataloader, val_dataloader,
    early_stop_step=2, K=4, annealing=False, T=1, n_groups=4, seed=0
):
    torch.manual_seed(seed)
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    layers_per_group = n_layers // n_groups

    num_to_unmask = K * layers_per_group

    new_head_mask = torch.rand(n_layers, n_heads).to(args.device)
    new_head_mask[new_head_mask>=0.5] = 1.0
    new_head_mask[new_head_mask<0.5] = 0.0
    
    val_scores = []
    eval_scores = []
    sparsities = []
    all_head_masks = []

    step = 0
    T = 1

    while step <= early_stop_step:
        # Store results and prepare for next run
        head_importance = torch.rand(n_layers, n_heads).to(args.device)
        val_score = evaluate(args, model, val_dataloader, head_mask=new_head_mask.clone())
        eval_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

        sparsity = 100 - new_head_mask.sum() / new_head_mask.numel() * 100
        logger.info(
            "Step %d: validation score: %f, evaluation score: %f, remaning heads %d (%.1f percents)",
            step,
            val_score,
            eval_score,
            new_head_mask.sum(),
            100 - sparsity,
        )

        val_scores.append(val_score * 100)
        eval_scores.append(eval_score * 100)
        sparsities.append(torch.round(sparsity))
        all_head_masks.append(new_head_mask.data)

        # Start pruning
        head_mask = new_head_mask.clone()  # save current head mask

        # if step < n_groups:
        #     group = step
        # else:
        #     group = random.randint(0,n_groups-1)
        group = step % n_groups

        indices = list(range(group*layers_per_group, (group+1)*layers_per_group))
        if not annealing:
            current_heads_to_unmask = head_importance[indices].view(-1).sort(descending = True)[1]
            current_heads_to_unmask = current_heads_to_unmask[:num_to_unmask] 
            current_heads_to_unmask += group * layers_per_group * n_heads
            logger.info("Heads to unmask for group %d: %s", group, str(current_heads_to_unmask.tolist()))
        else:
            lambdas = np.float128(head_importance[indices].view(-1).double().cpu().numpy())
            log_lambdas = np.log(lambdas) * (1/T)
            current_heads_to_unmask = log_sample_k_dpp(log_lambdas, num_to_unmask, seed)
            current_heads_to_unmask = [h + group * layers_per_group * n_heads for h in current_heads_to_unmask]
            logger.info("Heads to unmask for group %d: %s", group, str(current_heads_to_unmask))
        new_head_mask[indices] = 0.0
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_unmask] = 1.0
        new_head_mask = new_head_mask.view_as(head_mask)

        step += 1

        if T <= 1e-5:
            multiplicant = 1
        elif T <= 1000:
            multiplicant = 0.1

        if step % 4 == 0:
            T *= multiplicant  

    # file_name = "gibbs_per_group" + "_deterministic" if deterministic else ""
    # save_results(args, val_scores, sparsities, all_head_masks, file_name)

    val_scores = np.array(val_scores)
    desired_sparsity = np.round(100 - K / n_heads * 100)
    best_head_mask = None
    best_score = None
    i = None
    while True:
        if i is not None and i == np.argmax(val_scores):
            logger.info("Not converged to desired sparsity yet")
            break
        else:
            i = np.argmax(val_scores)
        if sparsities[i] == desired_sparsity:
            best_score = eval_scores[i]
            best_head_mask = all_head_masks[i]
            logger.info("Best score from iteration %d: %f", i, best_score)
            logger.info("Best head mask")
            print_2d_tensor(best_head_mask)
            # np.save(os.path.join(args.output_dir, str(K) + '_best_mask_per_layer.npy'), best_head_mask.cpu())
            break
        else:
            val_scores[i] = float("-inf")
    
    logger.info("Average score: %f", np.mean(eval_scores))

    return best_score, desired_sparsity, best_head_mask

def gibbs_sampling_test(    
    args, model, train_dataloader, eval_dataloader, val_dataloader=None,
    early_stop_step=2, K=4, annealing=False, T=1, n_groups=4
):
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads

    layers_per_group = n_layers // n_groups

    # num_to_unmask = K * layers_per_group

    new_head_mask = torch.rand(n_layers, n_heads).to(args.device)
    new_head_mask[new_head_mask>=0.5] = 1.0
    new_head_mask[new_head_mask<0.5] = 0.0
    
    val_scores = []
    eval_scores = []
    sparsities = []
    all_head_masks = []

    step = 0
    T = 1

    while step <= early_stop_step:
        # Store results and prepare for next run
        if val_dataloader is None:
            head_importance, val_score = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
        else:
            head_importance, _ = compute_heads_importance(
                args, model, train_dataloader, head_mask=new_head_mask.clone()
            )
            val_score = evaluate(args, model, val_dataloader, head_mask=new_head_mask.clone())
        eval_score = evaluate(args, model, eval_dataloader, head_mask=new_head_mask.clone())

        sparsity = 100 - new_head_mask.sum() / new_head_mask.numel() * 100
        logger.info(
            "Step %d: validation score: %f, evaluation score: %f, remaning heads %d (%.1f percents)",
            step,
            val_score,
            eval_score,
            new_head_mask.sum(),
            100 - sparsity,
        )

        val_scores.append(val_score * 100)
        eval_scores.append(eval_score * 100)
        sparsities.append(torch.round(sparsity))
        all_head_masks.append(new_head_mask.data)

        # Start pruning
        head_mask = new_head_mask.clone()  # save current head mask

        # if step < n_groups:
        #     group = step
        # else:
        #     group = random.randint(0,n_groups-1)
        group = step % n_groups
        num_to_unmask = K[group]
        indices = list(range(group*layers_per_group, (group+1)*layers_per_group))
        if not annealing:
            current_heads_to_unmask = head_importance[indices].view(-1).sort(descending = True)[1]
            current_heads_to_unmask = current_heads_to_unmask[:num_to_unmask] 
            current_heads_to_unmask += group * layers_per_group * n_heads
            logger.info("Heads to unmask for group %d: %s", group, str(current_heads_to_unmask.tolist()))
        else:
            lambdas = np.float128(head_importance[indices].view(-1).double().cpu().numpy())
            log_lambdas = np.log(lambdas) * (1/T)
            current_heads_to_unmask = log_sample_k_dpp(log_lambdas, num_to_unmask)
            current_heads_to_unmask = [h + group * layers_per_group * n_heads for h in current_heads_to_unmask]
            logger.info("Heads to unmask for group %d: %s", group, str(current_heads_to_unmask))
        new_head_mask[indices] = 0.0
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_unmask] = 1.0
        new_head_mask = new_head_mask.view_as(head_mask)

        step += 1

        if T <= 1e-5:
            multiplicant = 1
        elif T <= 1000:
            multiplicant = 0.1

        if step % 4 == 0:
            T *= multiplicant  

    # file_name = "gibbs_per_group" + "_deterministic" if deterministic else ""
    # save_results(args, val_scores, sparsities, all_head_masks, file_name)

    val_scores = np.array(val_scores)
    desired_sparsity = np.round(100 - sum(K) / (n_heads * n_layers) * 100)
    best_head_mask = None
    best_score = None
    i = None
    while True:
        if i is not None and i == np.argmax(val_scores):
            logger.info("Not converged to desired sparsity yet")
            break
        else:
            i = np.argmax(val_scores)
        if sparsities[i] == desired_sparsity:
            best_score = eval_scores[i]
            best_head_mask = all_head_masks[i]
            logger.info("Best score from iteration %d: %f", i, best_score)
            logger.info("Best head mask")
            print_2d_tensor(best_head_mask)
            # np.save(os.path.join(args.output_dir, str(K) + '_best_mask_per_layer.npy'), best_head_mask.cpu())
            break
        else:
            val_scores[i] = float("-inf")

    return best_score, desired_sparsity, best_head_mask

def plot_graph(args):
    for f in os.listdir(args.output_dir):
        if "_scores.npy" in f:
            file_path = os.path.join(args.output_dir, f)
            file_name = file_path[file_path.rfind('/')+1:file_path.rfind('_')]
            
            scores = np.load(file_path)
            sparsities = np.load(file_path.replace("scores", "sparsities"), allow_pickle=True)
            area = (auc(sparsities, scores) * 0.0001).cpu().numpy()
            plt.plot(sparsities, scores, marker='o', label=file_name + ": {:.4f}".format(area))

    plt.title("sparsity vs. accuracy")
    plt.xlabel("sparsity (%)")
    plt.ylabel("accuracy (%)")

    plt.xticks(np.arange(0,110,10))
    plt.yticks(np.arange(0,110,10))

    plt.grid(True)
    plt.legend()
    plt.savefig("results.png")

def save_results(args, scores, sparsities, all_head_masks, file_name):
    np.save(os.path.join(args.output_dir, file_name+'_scores.npy'), scores)
    np.save(os.path.join(args.output_dir, file_name+'_sparsities.npy'), sparsities)
    np.save(os.path.join(args.output_dir, file_name+'_masks.npy'), all_head_masks)

def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))