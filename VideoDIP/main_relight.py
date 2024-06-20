import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from video_dip.callbacks.image_logger import ImageLogger
from video_dip.models.modules.relight import RelightVDPModule
from video_dip.data.datamodule import VideoDIPDataModule
from video_dip.models.optical_flow import RAFT, RAFTModelSize, Farneback
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

def relight(
    learning_rate=2e-3,
    loss_weights=[1, .02],
    milestones=[5, 15, 45, 75],
    gamma=.5,
    warmup=True,
    input_path="datasets/relighting/outdoor_png/input/pair76",
    target_path="datasets/relighting/outdoor_png/GT/pair76",
    flow_path="flow_outputs",
    batch_size=4,
    num_workers=4,
    max_epochs=100,
    devices=[1],
    logger='tb',
    flow_model='raft',
    early_stopping=True
):
    # Initialize the model
    model = RelightVDPModule(
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

    if flow_model == 'raft':
        data_module.dump_optical_flow(flow_model=RAFT(RAFTModelSize.LARGE))
    elif flow_model == 'farneback':
        data_module.dump_optical_flow(flow_model=Farneback())
    else:
        raise ValueError(f"Invalid flow model: {flow_model}")

    # Initialize the loggers
    if logger == 'tb':
        logger = TensorBoardLogger("tb_logs", name=f"video_dip_relight_{input_path.split('/')[-1]}")
    elif logger == 'wandb':
        logger = WandbLogger(project="video_dip_relight")
        logger.watch(model)
    else:
        raise ValueError(f"Invalid logger: {logger}")
    
    callbacks = [
        ImageLogger(num_images=1),
        LearningRateMonitor(logging_interval='epoch')  # Log learning rate at every training step
    ]
    if early_stopping:
        callbacks.append(EarlyStopping(monitor='psnr', patience=20, mode='max'))

    # Initialize the trainer with the logger
    trainer = pl.Trainer(
        logger=logger, 
        devices=devices, 
        max_epochs=max_epochs, 
        callbacks=callbacks,
        benchmark=True,
        num_sanity_val_steps=0
    )

    # Fit the model
    trainer.fit(model, datamodule=data_module)

    return trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Relight video using RelightVDPModule.")
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="Learning rate for the model.")
    parser.add_argument("--loss_weights", nargs=2, type=float, default=[1, .02], help="Loss weights for the model.")
    parser.add_argument("--milestones", nargs='+', type=int, default=[5, 15, 45, 75], help="Milestones for learning rate scheduling.")
    parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for learning rate scheduling.")
    parser.add_argument("--input_path", type=str, default="datasets/relight/input/pair76", help="Input path for the relighting videos.")
    parser.add_argument("--target_path", type=str, default="datasets/relight/GT/pair76", help="Target path for the ground truth videos.")
    parser.add_argument("--flow_path", type=str, default="flow_outputs", help="Path to save the optical flow outputs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for the data loader.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the data loader.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs for training.")
    parser.add_argument("--devices", nargs='+', type=int, default=[1], help="Devices to use for training.")
    parser.add_argument("--logger", type=str, choices=['tb', 'wandb'], default='tb', help="Logger to use for training.")
    parser.add_argument("--flow_model", type=str, choices=['raft', 'farneback'], default='raft', help="Optical flow model to use.")
    parser.add_argument("--warmup", type=bool, default=True, help="Warmup flag for the model.")
    parser.add_argument("--early_stopping", type=bool, default=True, help="Early stopping flag for the model.")

    args = parser.parse_args()
    
    relight(
        learning_rate=args.learning_rate,
        loss_weights=args.loss_weights,
        milestones=args.milestones,
        gamma=args.gamma,
        input_path=args.input_path,
        target_path=args.target_path,
        flow_path=args.flow_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        devices=args.devices,
        logger=args.logger,
        flow_model=args.flow_model
    )
