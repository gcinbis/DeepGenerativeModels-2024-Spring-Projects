import os
import argparse
import numpy as np
from pathlib import Path
from main_relight import relight  # Import the relighting function from the provided script

def benchmark_relight(dataset_path, **relight_kwargs):
    # Discover all sub-folders within the dataset folder
    sub_folders = [f.path for f in os.scandir(os.path.join(dataset_path, 'GT')) if f.is_dir()]

    print(f"Found {len(sub_folders)} sub-folders in the dataset.")

    all_metrics = []

    with open('relight_metrics.txt', 'w') as f:
        f.write('====================================================================================================\n')
        f.write(f"Found {len(sub_folders)} sub-folders in the dataset.\n")
        f.write('====================================================================================================\n')

    for sub_folder in sub_folders:
        print('====================================================================================================')
        print(f"Processing sub-folder: {sub_folder}")
        print('====================================================================================================')

        # Run the relighting script for each sub-folder
        metrics = relight(
            input_path=sub_folder.replace('GT', 'input'),
            target_path=sub_folder,
            **relight_kwargs
        )

        all_metrics.append(metrics)

        with open('relight_metrics.txt', 'a') as f:
            f.write(f"Metrics for sub-folder: {sub_folder}\n")
            f.write(f"{metrics}\n")

    # Average the metrics
    sums = {}
    counts = {}

    # Loop through all metrics and sum up the values for each field
    for metrics in all_metrics:
        for metric in metrics:
            for key, value in metric.items():
                if key not in sums:
                    sums[key] = 0
                    counts[key] = 0
                sums[key] += value
                counts[key] += 1

    # Calculate averages
    averaged_metrics = {key: sums[key] / counts[key] for key in sums}
    print("All Metrics:", all_metrics)

    with open('relight_metrics.txt', 'a') as f:
        f.write(f"Averaged Metrics: {averaged_metrics}\n")
    
    return averaged_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark relighting on multiple sub-folders.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset folder containing sub-folders.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for the model.")
    parser.add_argument("--loss_weights", nargs=2, type=float, default=[1, .02], help="Loss weights for the model.")
    parser.add_argument("--milestones", nargs='+', type=int, default=[3, 5, 45, 75], help="Milestones for learning rate scheduling.")
    parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for learning rate scheduling.")
    parser.add_argument("--warmup", type=bool, default=True, help="Warmup flag for the model.")
    parser.add_argument("--flow_path", type=str, default="flow_outputs", help="Path to save the optical flow outputs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for the data loader.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the data loader.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs for training.")
    parser.add_argument("--devices", nargs='+', type=int, default=[1], help="Devices to use for training.")
    parser.add_argument("--logger", type=str, choices=['tb', 'wandb'], default='tb', help="Logger to use for training.")
    parser.add_argument("--flow_model", type=str, choices=['raft', 'farneback'], default='raft', help="Optical flow model to use.")

    args = parser.parse_args()
    
    relight_kwargs = {
        'learning_rate': args.learning_rate,
        'loss_weights': args.loss_weights,
        'milestones': args.milestones,
        'gamma': args.gamma,
        'warmup': args.warmup,
        'flow_path': args.flow_path,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'max_epochs': args.max_epochs,
        'devices': args.devices,
        'logger': args.logger,
        'flow_model': args.flow_model
    }
    
    results = benchmark_relight(args.dataset_path, **relight_kwargs)
    print("Averaged Metrics:", results)
