import os, sys
import argparse
import math
from tqdm import tqdm
import json

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import timm, timm.data

from ttfsformer.model import vit, eva, set_monitor
from ttfsformer.util import AverageMeter, accuracy, imagenet_loader, load_weights, get_monitor_data
from ttfsformer.util.monitor import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/hpc/zls/data/imagenet-1k/val")
    parser.add_argument("--weight", type=str, default="weight/vit_small_patch16_224.augreg_in21k_ft_in1k")
    parser.add_argument("--batch_size", type=int, default=50)

    args = parser.parse_args()

    with open(os.path.join(args.weight, "config.json")) as file:
        model_name = json.load(file)["architecture"]

    model = timm.create_model(model_name, pretrained=False)
    load_weights(args.weight, model)
    data_loader = imagenet_loader(args.path, args.weight, args.batch_size)

    set_monitor(model, lambda: MinMaxMonitor())
    model.cuda()
    model.eval()

    print(model)

    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc=f"Eval {model_name} on ImageNet") as t:
            for images, labels in data_loader:
                images, labels = images.cuda(), labels.cuda()
                # compute output
                output = model(images)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

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

    print(f"Original {model_name} Top1 Acc={top1.mean:.5f}, Top5 Acc={top5.mean:.5f}")

    data = get_monitor_data(model)
    os.makedirs(f"data/{model_name}/", exist_ok=True)
    with open(f"data/{model_name}/monitor-100.json", "w") as file:
        json.dump(data, file)


if __name__ == "__main__":
    main()
