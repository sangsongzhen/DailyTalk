import argparse
import os
import numpy as np
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import get_variance_level, to_device, log, synth_one_sample
# from utils.tools import get_variance_level, log, synth_one_sample
from model import CompTransTTSLoss
from dataset import Dataset


# def to_device(data, device):
#     if isinstance(data, (list, tuple)):
#         return type(data)(to_device(x, device) for x in data)  # 保持原始类型（List/Tuple）
#     elif isinstance(data, torch.Tensor):
#         return data.to(device)
#     elif isinstance(data, np.ndarray):
#         return torch.from_numpy(data).to(device)
#     else:
#         return data


def evaluate(device, model, step, configs, logger=None, vocoder=None, losses=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    level_tag, *_ = get_variance_level(preprocess_config, model_config)
    dataset = Dataset(
        "val_{}.txt".format(level_tag), preprocess_config, model_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = CompTransTTSLoss(preprocess_config, model_config, train_config).to(device)

    # Evaluation
    loss_sums = [{k:0 for k in loss.keys()} if isinstance(loss, dict) else 0 for loss in losses]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]), step=step)
                batch[9:11], output = output[-2:], output[:-2] # Update pitch and energy level

                # Cal Loss
                losses = Loss(batch, output, step=step)

                for i in range(len(losses)):
                    if isinstance(losses[i], dict):
                        for k in loss_sums[i].keys():
                            loss_sums[i][k] += losses[i][k].item() * len(batch[0])
                    else:
                        loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = []
    loss_means_ = []
    for loss_sum in loss_sums:
        if isinstance(loss_sum, dict):
            loss_mean = {k:v / len(dataset) for k, v in loss_sum.items()}
            loss_means.append(loss_mean)
            loss_means_.append(sum(loss_mean.values()))
        else:
            loss_means.append(loss_sum / len(dataset))
            loss_means_.append(loss_sum / len(dataset))

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, CTC Loss: {:.4f}, Binarization Loss: {:.4f}".format(
        *([step] + [l for l in loss_means_])
    )

    if logger is not None:
        fig, fig_attn, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        if fig_attn is not None:
            log(
                logger,
                img=fig_attn,
                tag="Validation_attn/step_{}_{}".format(step, tag),
            )
        log(
            logger,
            img=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message
