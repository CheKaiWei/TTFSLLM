import json

import torch, torch.utils.data
from torchvision import datasets, transforms
import timm.data


def imagenet_loader(
    path: str,
    weight_dir: str,
    batch_size: int,
    num_workers: int = 0,
    num_shards: int = 1,
    shard_index: int = 0,
) -> torch.utils.data.DataLoader:
    if num_shards < 1:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")

    with open(f"{weight_dir}/config.json", "r") as file:
        pretrained_cfg = json.load(file)["pretrained_cfg"]
    config = timm.data.resolve_data_config(pretrained_cfg=pretrained_cfg)
    transforms = timm.data.create_transform(**config, is_training=False)
    print(transforms)
    dataset = datasets.ImageFolder(path, transforms)
    if num_shards > 1:
        start = len(dataset) * shard_index // num_shards
        end = len(dataset) * (shard_index + 1) // num_shards
        dataset = torch.utils.data.Subset(dataset, range(start, end))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
    )
