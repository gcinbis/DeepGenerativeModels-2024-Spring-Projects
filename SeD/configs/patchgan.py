import torch
from pytorch_lightning.strategies import DDPStrategy



accelerator = 'gpu'
device = torch.device("cuda") if accelerator=="gpu" else torch.device("cpu")
if accelerator == 'cpu':
    pl_trainer = dict(max_epochs=1000, accelerator=accelerator, log_every_n_steps=50, strategy=DDPStrategy(find_unused_parameters=True), devices=1, sync_batchnorm=True) # CHECK sync_batchnorm in this and below part !!!
else:
    pl_trainer = dict(max_epochs=1000, accelerator=accelerator, log_every_n_steps=50, strategy=DDPStrategy(find_unused_parameters=True), devices=torch.cuda.device_count(), sync_batchnorm=True)  # CHECK strategy and find_unused_parameters!!!

train_batch_size = 16
val_batch_size = 8
test_batch_size = 8

image_size = 256


###########################
##### Dataset Configs #####
###########################

dataset_module = dict(
    num_workers=4,
    train_batch_size=train_batch_size,
    val_batch_size=val_batch_size,
    test_batch_size=test_batch_size,
    train_dataset_config=dict(image_size=256, image_dir_hr="data/dataset_cropped/hr", image_dir_lr="data/dataset_cropped/lr", downsample_factor=4),
    val_dataset_config=dict(image_size=256, image_dir_hr="data/evaluation/hr/manga109", image_dir_lr="data/evaluation/lr/manga109"),
    test_dataset_config=dict(image_size=256, image_dir_hr="data/evaluation/hr/manga109", image_dir_lr="data/evaluation/lr/manga109"),
)

##################
##### Losses #####
##################
vgg_ckpt_path="pretrained_models/vgg16.pth"
loss_dict = dict(
    VGG=dict(weight=1000.0, model_config=dict(path=vgg_ckpt_path, output_layer_idx=23, resize_input=False)),
    Adversarial_G=dict(weight=1.0),
    MSE=dict(weight=1e-1),
    Adversarial_D=dict(r1_gamma=10.0, r2_gamma=0.0)
)

#########################
##### Model Configs #####
#########################

super_resolution_module_config = dict(loss_dict=loss_dict, 
    generator_learning_rate=1e-4, discriminator_learning_rate=1e-5, 
    generator_decay_steps=[50_000, 100_000, 150_000, 200_000, 250_000], 
    discriminator_decay_steps=[50_000, 100_000, 150_000, 200_000, 250_000], 
    generator_decay_gamma=0.5, discriminator_decay_gamma=0.5,
    clip_generator_outputs=True,
    use_sed_discriminator=False)

#######################
###### Callbacks ######
#######################

ckpt_callback = dict(every_n_train_steps=4000, save_top_k=1, save_last=True, monitor='lpips_test', mode='min')
synthesize_callback_train = dict(num_samples=12, eval_every=2000) # TODO: 4000
synthesize_callback_test = dict(num_samples=6, eval_every=2000)
fid_callback = dict(eval_every=2000)
lpips_callback = dict(eval_every=2000)
