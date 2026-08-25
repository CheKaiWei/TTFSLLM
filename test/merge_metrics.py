import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metrics = []
    for path in args.inputs:
        with open(path) as file:
            metrics.append(json.load(file))

    architecture = metrics[0]["architecture"]
    variant = metrics[0].get("variant")
    if any(item["architecture"] != architecture or item.get("variant") != variant for item in metrics):
        raise ValueError("All metric shards must have the same architecture and variant")
    monitor_input = metrics[0].get("monitor_input")
    if any(item.get("monitor_input") != monitor_input for item in metrics):
        raise ValueError("All metric shards must use the same monitor input")
    expected_num_shards = metrics[0]["num_shards"]
    if len(metrics) != expected_num_shards or any(item["num_shards"] != expected_num_shards for item in metrics):
        raise ValueError(f"Expected {expected_num_shards} metric shards, got {len(metrics)}")
    shard_indices = [item["shard_index"] for item in metrics]
    if set(shard_indices) != set(range(expected_num_shards)):
        raise ValueError(f"Shard indices must cover [0, {expected_num_shards}), got {sorted(shard_indices)}")

    total_samples = sum(item["samples"] for item in metrics)
    merged = {
        "architecture": architecture,
        "variant": variant,
        "precision": metrics[0].get("precision"),
        "noise": metrics[0].get("noise"),
        "monitor_input": monitor_input,
        "num_shards": expected_num_shards,
        "samples": total_samples,
        "batches": sum(item["batches"] for item in metrics),
        "top1": sum(item["top1"] * item["samples"] for item in metrics) / total_samples,
        "top5": sum(item["top5"] * item["samples"] for item in metrics) / total_samples,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as file:
        json.dump(merged, file, indent=2)
    print(
        f"Merged {len(metrics)} shards for {architecture} ({variant}): "
        f"samples={total_samples}, top1={merged['top1']:.5f}, top5={merged['top5']:.5f}"
    )


if __name__ == "__main__":
    main()
