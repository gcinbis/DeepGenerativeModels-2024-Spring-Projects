import os
import argparse
import numpy as np
from pathlib import Path
from main_segmentation import segment  # Import the segmentation function from the provided script

def benchmark_segment(dataset_path, **segment_kwargs):
    # Discover all sub-folders within the dataset folder
    sub_folders = [f.path for f in os.scandir(os.path.join(dataset_path, 'GT')) if f.is_dir()]
    flow_path = segment_kwargs["flow_path"]

    print(f"Found {len(sub_folders)} sub-folders in the dataset.")

    all_metrics = []

    for sub_folder in sub_folders:
        print('====================================================================================================')
        print(f"Processing sub-folder: {sub_folder}")
        print('====================================================================================================')


        experiment_name = sub_folder.split('/')[-1]
        flow_subfolder = os.path.join(flow_path + experiment_name)
        segment_kwargs['flow_path'] = flow_subfolder    
        os.makedirs(flow_subfolder, exist_ok=True)

        # Run the segmentation script for each sub-folder
        metrics = segment(
            input_path=sub_folder.replace('GT', 'input'),
            target_path=sub_folder,
            **segment_kwargs
        )

        all_metrics.append(metrics)

    # Average the metrics
    averaged_metrics = {key: np.mean([metric[key] for metric in all_metrics]) for key in all_metrics[0]}

    print("All Metrics:", all_metrics)
    
    return averaged_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark segmentation on multiple sub-folders.")
    parser.add_argument("--dataset_path", type=str, default="/home/alpfischer/METU-Courses/VideoDIP/datasets/benchmark_root", help="Path to the dataset folder containing sub-folders.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the model.")
    parser.add_argument("--loss_weights", nargs=5, type=float, default=[.001, 1, 1, .001, .01], help="Loss weights for the model.")
    parser.add_argument("--milestones", nargs='+', type=int, default=[5, 15, 45, 75], help="Milestones for learning rate scheduling.")
    parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for learning rate scheduling.")
    parser.add_argument("--warmup", type=bool, default=True, help="Warmup flag for the model.")
    parser.add_argument("--flow_path", type=str, default="/home/alpfischer/Downloads/DAVIS-data/DAVIS/Annotations/flows/", help="Path to save the optical flow outputs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for the data loader.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for the data loader.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs for training.")
    parser.add_argument("--devices", nargs='+', type=int, default=[0], help="Devices to use for training.")
    parser.add_argument("--logger", type=str, choices=['tb', 'wandb'], default='wandb', help="Logger to use for training.")

    args = parser.parse_args()
    
    segment_kwargs = {
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
        'logger': args.logger
    }
    
    results = benchmark_segment(args.dataset_path, **segment_kwargs)
    print("Averaged Metrics:", results)
