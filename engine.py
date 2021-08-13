# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train and eval functions used in main.py

Mostly copy-paste from https://github.com/facebookresearch/deit/blob/main/engine.py
"""

import math
from typing import Iterable, Optional
import torch

from timm.data import Mixup
from timm.utils import accuracy
from einops import rearrange

import utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    world_size: int = 1, distributed: bool = True, amp=True,
                    finetune=False
                    ):
    if finetune:
        model.train(not finetune)
    else:
        model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        batch_size = targets.size(0)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                raise ValueError("Loss is {}, stopping training".format(loss_value))

            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

            if amp:
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
            else:
                loss.backward(create_graph=is_second_order)
                if max_norm is not None and max_norm != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, world_size, distributed=True, amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    outputs = []
    targets = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast(enabled=amp):
            output = model(images)

        if distributed:
            outputs.append(concat_all_gather(output))
            targets.append(concat_all_gather(target))
        else:
            outputs.append(output)
            targets.append(target)

    num_data = len(data_loader.dataset)
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    real_acc1, real_acc5 = accuracy(outputs[:num_data], targets[:num_data], topk=(1, 5))
    real_loss = criterion(outputs, targets)
    metric_logger.update(loss=real_loss.item())
    metric_logger.meters['acc1'].update(real_acc1.item())
    metric_logger.meters['acc5'].update(real_acc5.item())
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor.contiguous(), async_op=False)

    if tensor.dim() == 1:
        output = rearrange(tensors_gather, 'n b -> (b n)')
    else:
        output = rearrange(tensors_gather, 'n b c -> (b n) c')
    return output
