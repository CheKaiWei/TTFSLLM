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
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--monitor_input", type=str)
    parser.add_argument("--metrics_output", type=str)
    parser.add_argument("--no_print_model", dest="print_model", action="store_false")

    args = parser.parse_args()

    with open(os.path.join(args.weight, "config.json")) as file:
        model_name = json.load(file)["architecture"]

    model = timm.create_model(model_name, pretrained=False)
    load_weights(args.weight, model)
    data_loader = imagenet_loader(
        args.path,
        args.weight,
        args.batch_size,
        args.num_workers,
        args.num_shards,
        args.shard_index,
    )

    model.cuda()
    model.eval()

    monitor_input = args.monitor_input or f"data/{model_name}/monitor-100.json"
    with open(monitor_input, "r") as file:
        param = SpikeParamDict(json.load(file))

    config = SpikeConfig(precision=args.precision, noise=args.noise)
    if isinstance(model, vit.VisionTransformer):
        model = SpikingVisionTransformer(model, config, param)
    elif isinstance(model, eva.Eva):
        model = SpikingEva(model, config, param)
    else:
        raise NotImplementedError(f"Model {type(model)} not implemented")

    if args.print_model:
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

    output_dir = f"data/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    if args.precision is not None:
        variant = f"precision-{args.precision}"
    elif args.noise is not None:
        variant = f"noise-{args.noise:g}"
    else:
        variant = "precise"
    shard_suffix = "" if args.num_shards == 1 else f".shard-{args.shard_index}-of-{args.num_shards}"
    metrics_output = args.metrics_output or f"{output_dir}/ttfs-{variant}-metrics{shard_suffix}.json"
    metrics = {
        "architecture": model_name,
        "variant": variant,
        "precision": args.precision,
        "noise": args.noise,
        "monitor_input": monitor_input,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "samples": len(data_loader.dataset),
        "batches": len(data_loader),
        "top1": top1.mean,
        "top5": top5.mean,
    }
    os.makedirs(os.path.dirname(metrics_output) or ".", exist_ok=True)
    with open(metrics_output, "w") as file:
        json.dump(metrics, file, indent=2)


if __name__ == "__main__":
    main()
