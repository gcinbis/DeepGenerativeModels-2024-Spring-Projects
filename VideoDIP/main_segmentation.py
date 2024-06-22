import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from video_dip.callbacks.image_logger import ImageLogger
from video_dip.models.modules.segmentation import SegmentationVDPModule
from video_dip.data.datamodule import VideoDIPDataModule
from video_dip.models.optical_flow.raft import RAFT, RAFTModelSize
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging

def segment(
    learning_rate=1e-3,
    loss_weights=[.001, 1, 1, .001, .01],
    milestones=[5, 15, 45, 75],
    gamma=.5,
    warmup=True,
    input_path="datasets/input/bear",
    target_path="datasets/GT/pair1",
    flow_path="flow_outputs",
    batch_size=2,
    num_workers=8,
    max_epochs=100,
    devices=[0],
    logger='tb'
):
    # Initialize the model
    model = SegmentationVDPModule(
        learning_rate=learning_rate, 
        loss_weights=loss_weights,
        multi_step_scheduling_kwargs={
            'milestones': milestones,
            'gamma': gamma,
        },
        warmup=warmup
    )

    # Initialize the data module
    data_module = VideoDIPDataModule(
        input_path=input_path, 
        target_path=target_path,
        flow_path=flow_path,
        batch_size=batch_size, 
        num_workers=num_workers
    )

    data_module.dump_optical_flow(flow_model=RAFT(RAFTModelSize.LARGE))
 
    # Initialize the loggers
    if logger == 'tb':
        logger = TensorBoardLogger("tb_logs", name="my_model")
    elif logger == 'wandb':
        logger = WandbLogger(project="video_dip_segmentation")
        logger.watch(model)
    else:
        raise ValueError(f"Invalid logger: {logger}")

    # Initialize the trainer with the logger
    trainer = pl.Trainer(
        logger=logger, 
        devices=devices, 
        max_epochs=max_epochs, 
        callbacks=[
            ImageLogger(num_images=1),
            LearningRateMonitor(logging_interval='epoch')  # Log learning rate at every training step
        ],
        benchmark=True,
    )

    # Fit the model
    trainer.fit(model, datamodule=data_module)

    return trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segment video using SegmentationVDPModule.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the model.")
    parser.add_argument("--loss_weights", nargs=5, type=float, default=[.001, 1, 1, .001, .01], help="Loss weights for the model.")
    parser.add_argument("--milestones", nargs='+', type=int, default=[5, 15, 45, 75], help="Milestones for learning rate scheduling.")
    parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for learning rate scheduling.")
    parser.add_argument("--warmup", type=bool, default=True, help="Warmup flag for the model.")
    parser.add_argument("--input_path", type=str, default="datasets/input/blackswan", help="Input path for the segmentation videos.")
    parser.add_argument("--target_path", type=str, default="datasets/GT/blackswan", help="Target path for the ground truth videos.")
    parser.add_argument("--flow_path", type=str, default="datasets/input/blackswan_flow", help="Path to save the optical flow outputs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for the data loader.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for the data loader.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs for training.")
    parser.add_argument("--devices", nargs='+', type=int, default=[0], help="Devices to use for training.")
    parser.add_argument("--logger", type=str, choices=['tb', 'wandb'], default='wandb', help="Logger to use for training.")

    args = parser.parse_args()
    
    segment(
        learning_rate=args.learning_rate,
        loss_weights=args.loss_weights,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup=args.warmup,
        input_path=args.input_path,
        target_path=args.target_path,
        flow_path=args.flow_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        devices=args.devices,
        logger=args.logger
    )
