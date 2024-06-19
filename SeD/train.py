import pytorch_lightning as pl
from datasets.dataset_module import DatasetModule
from models.super_resolution_module import SuperResolutionModule
from argparse import ArgumentParser
from utils.config_utils import parse_config
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from callbacks.logger.image_logger import ImageLoggerCallback
from callbacks.eval.lpips import LPIPSCallback
from pytorch_lightning.utilities import rank_zero_only
import os
import shutil
import torch
import numpy as np
import random
from datetime import datetime
from pytorch_lightning.strategies import DDPStrategy


# import warnings
# warnings.filterwarnings('ignore')

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

@rank_zero_only
def log_config_file(config):
    os.makedirs(config.log_path, exist_ok=True)
    shutil.copyfile(args.config_file, os.path.join(config.log_path, "config.py"))


def extend_config_parameters(config, is_debug):
    experiment_name = args.config_file.split("/")[-1].split(".")[0]
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f'{date_time}_{experiment_name}'
    config["experiment_name"] = experiment_name
    if is_debug:
        config["log_path"] = os.path.join('logs', '.debug', experiment_name)
    else:
        config["log_path"] = os.path.join('logs', experiment_name)
    config.ckpt_callback["dirpath"] = os.path.join(config["log_path"], 'checkpoint')

def train(args):
    config = parse_config(args.config_file)
    extend_config_parameters(config, is_debug=args.debug)
    model = SuperResolutionModule(**config.super_resolution_module_config)
    
    if args.device == "gpu":
        accelerator = 'gpu'
        device = torch.device("cuda")
        pl_trainer = dict(max_epochs=1000, accelerator=accelerator, log_every_n_steps=50, strategy=DDPStrategy(find_unused_parameters=True), devices=torch.cuda.device_count(), sync_batchnorm=True)  
    else:
        device = torch.device("cpu")
        pl_trainer = dict(max_epochs=1000, accelerator='cpu', log_every_n_steps=50, strategy=DDPStrategy(find_unused_parameters=True), devices=1) # CHECK sync_batchnorm in this and below part !!!


    ckpt_path = None
    if args.resume_from is not None:
        ckpt_path = os.path.join(args.resume_from, "checkpoint", "last.ckpt")
        config.log_path = args.resume_from
        config.ckpt_callback.dirpath = os.path.join(config.log_path, 'checkpoint')
            
    log_config_file(config)

    data_module = DatasetModule(**config.dataset_module)
    data_module.setup('training')
    data_module.setup('test')

    train_dataloader = data_module.train_dataloader()
    if not args.debug:
        test_dataloader = data_module.test_dataloader()

    csv_logger = pl_loggers.CSVLogger(save_dir=config.log_path, flush_logs_every_n_steps=50)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=config.log_path+"/tensorboard")

    
    ckpt_callback = ModelCheckpoint(**config.ckpt_callback)
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
    synthesize_callback_train = ImageLoggerCallback(data_module.train_dataset, "training", **config.synthesize_callback_train)
    synthesize_callback_test = ImageLoggerCallback(data_module.test_dataset, "test", **config.synthesize_callback_test)
    
    if not args.debug:
        lpips_callback_test = LPIPSCallback(test_dataloader, dataset_type="test", **config.lpips_callback)

    if not args.debug:
        trainer = pl.Trainer(logger=[csv_logger, tb_logger], 
                            callbacks=[ lpips_callback_test, 
                                        synthesize_callback_train, 
                                        synthesize_callback_test, lr_monitor_callback,
                                        ckpt_callback], 
                            **pl_trainer)
    else:
        trainer = pl.Trainer(logger=[csv_logger, tb_logger], 
                            callbacks=[synthesize_callback_train, 
                                        synthesize_callback_test, 
                                        lr_monitor_callback, ckpt_callback], 
                            **pl_trainer)

    seed_all(seed=0)
    trainer.fit(model, train_dataloaders=train_dataloader, ckpt_path=ckpt_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, default='configs/default_config.py', help='Path to config file')
    parser.add_argument('--resume_from', type=str, default=None, help='Log folder of the model to be resumed')
    parser.add_argument('--device', type=str, default='cpu', help='device : cpu or gpu')
    parser.add_argument('--debug', action='store_true', help='Start in debugging mode')
    args = parser.parse_args()
    train(args)