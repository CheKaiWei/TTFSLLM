import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor_inputs", nargs="+", required=True)
    parser.add_argument("--metrics_inputs", nargs="+", required=True)
    parser.add_argument("--monitor_output", required=True)
    parser.add_argument("--metrics_output", required=True)
    parser.add_argument("--monitor_mode", choices=("range", "mean"), required=True)
    args = parser.parse_args()

    if len(args.monitor_inputs) != len(args.metrics_inputs):
        raise ValueError("monitor_inputs and metrics_inputs must contain the same number of shards")

    with open(args.monitor_inputs[0]) as file:
        merged_monitor = json.load(file)
    expected_keys = list(merged_monitor)
    expected_key_set = set(expected_keys)

    with open(args.metrics_inputs[0]) as file:
        first_metrics = json.load(file)
    all_metrics = [first_metrics]
    monitor_shards = [merged_monitor]

    for monitor_path, metrics_path in zip(args.monitor_inputs[1:], args.metrics_inputs[1:]):
        with open(monitor_path) as file:
            monitor = json.load(file)
        monitor_keys = set(monitor)
        if monitor_keys != expected_key_set:
            missing = sorted(expected_key_set - monitor_keys)
            extra = sorted(monitor_keys - expected_key_set)
            raise ValueError(f"Monitor keys differ for {monitor_path}: missing={missing}, extra={extra}")
        monitor_shards.append(monitor)
        with open(metrics_path) as file:
            all_metrics.append(json.load(file))

    architecture = first_metrics["architecture"]
    expected_num_shards = first_metrics["num_shards"]
    if any(metrics["architecture"] != architecture for metrics in all_metrics):
        raise ValueError("All metric shards must have the same architecture")
    if len(all_metrics) != expected_num_shards or any(
        metrics["num_shards"] != expected_num_shards for metrics in all_metrics
    ):
        raise ValueError(f"Expected {expected_num_shards} metric shards, got {len(all_metrics)}")
    shard_indices = [metrics["shard_index"] for metrics in all_metrics]
    if set(shard_indices) != set(range(expected_num_shards)):
        raise ValueError(f"Shard indices must cover [0, {expected_num_shards}), got {sorted(shard_indices)}")

    if args.monitor_mode == "range":
        merged_monitor = {
            key: [min(shard[key][0] for shard in monitor_shards), max(shard[key][1] for shard in monitor_shards)]
            for key in expected_keys
        }
    else:
        total_batches = sum(metrics["batches"] for metrics in all_metrics)
        merged_monitor = {
            key: [
                sum(shard[key][bound] * metrics["batches"] for shard, metrics in zip(monitor_shards, all_metrics))
                / total_batches
                for bound in (0, 1)
            ]
            for key in expected_keys
        }

    total_samples = sum(metrics["samples"] for metrics in all_metrics)
    merged_metrics = {
        "architecture": architecture,
        "num_shards": expected_num_shards,
        "samples": total_samples,
        "batches": sum(metrics["batches"] for metrics in all_metrics),
        "top1": sum(metrics["top1"] * metrics["samples"] for metrics in all_metrics) / total_samples,
        "top5": sum(metrics["top5"] * metrics["samples"] for metrics in all_metrics) / total_samples,
    }

    os.makedirs(os.path.dirname(args.monitor_output) or ".", exist_ok=True)
    with open(args.monitor_output, "w") as file:
        json.dump(merged_monitor, file)
    os.makedirs(os.path.dirname(args.metrics_output) or ".", exist_ok=True)
    with open(args.metrics_output, "w") as file:
        json.dump(merged_metrics, file, indent=2)

    print(
        f"Merged {len(all_metrics)} shards for {merged_metrics['architecture']}: "
        f"samples={total_samples}, top1={merged_metrics['top1']:.5f}, top5={merged_metrics['top5']:.5f}"
    )


if __name__ == "__main__":
    main()
