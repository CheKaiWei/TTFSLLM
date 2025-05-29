import os, sys
import argparse
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import timm, timm.data

from ttfsformer.util import AverageMeter, accuracy, imagenet_loader, load_weights
from ttfsformer.model import vit, eva
from ttfsformer.model.spiking_vit import SpikingVisionTransformer
from ttfsformer.model.spiking_eva import SpikingEva
from ttfsformer.spike import SpikeConfig, SpikeParamDict, value_to_ttfs, ttfs_to_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/hpc/zls/data/imagenet-1k/val")
    parser.add_argument("--weight", type=str, default="weight/vit_small_patch16_224.augreg_in21k_ft_in1k")
    parser.add_argument("--precision", type=int)
    parser.add_argument("--noise", type=float)
    parser.add_argument("--batch_size", type=int, default=50)

    args = parser.parse_args()

    with open(os.path.join(args.weight, "config.json")) as file:
        model_name = json.load(file)["architecture"]

    model = timm.create_model(model_name, pretrained=False)
    load_weights(args.weight, model)
    data_loader = imagenet_loader(args.path, args.weight, args.batch_size)

    model.cuda()
    model.eval()

    with open(f"data/{model_name}/monitor-100.json", "r") as file:
        param = SpikeParamDict(json.load(file))

    config = SpikeConfig(precision=args.precision, noise=args.noise)
    if isinstance(model, vit.VisionTransformer):
        model = SpikingVisionTransformer(model, config, param)
    elif isinstance(model, eva.Eva):
        model = SpikingEva(model, config, param)
    else:
        raise NotImplementedError(f"Model {type(model)} not implemented")

    print(model)

    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc=f"Eval {model_name} on ImageNet") as t:
            for images, labels in data_loader:
                images, labels = images.cuda(), labels.cuda()
                # compute output
                spike_input = value_to_ttfs(images, param.at("monitor_0"))
                spike_output = model(spike_input)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(spike_output, labels, topk=(1, 5))

                top1.update(acc1, images.size(0))
                top5.update(acc5, images.size(0))
                t.set_postfix(
                    {
                        "top1": top1.mean,
                        "top5": top5.mean,
                        "resolution": images.shape[-1],
                    }
                )
                t.update(1)

    print(f"Spiking {model_name}, Top1 Acc={top1.mean:.5f}, Top5 Acc={top5.mean:.5f}")


if __name__ == "__main__":
    main()
