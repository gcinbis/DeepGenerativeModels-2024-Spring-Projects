import torch

# Set the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# define the constants
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8
BATCH_SIZE = 1
root_dir = "../dataset/cc3m/train"

# training parameters
num_train_epochs = 6
Lambda = 1.0
save_steps = 1000

# optimizer parameters
learning_rate = 1e-5
discriminative_learning_rate = 1e-4  # New learning rate for discriminative tasks
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 1e-4
adam_epsilon = 1e-8
use_contrastive_loss = True

# IMAGE TO TEXT
test_dataset = "CIFAR10"  # Set to "CIFAR100" to use CIFAR100 dataset

# output directory
train_output_dir = "../results/output_1"
test_output_dir = "../results/" + test_dataset
inference_output_dir = "../results/text_to_image/output_1/last"

# Load the models
model_file = "data/v1-5-pruned.ckpt"
train_unet_file = None  # Set to None to finetune from scratch, if specified, the diffusion model will be loaded from this file
test_unet_file = "data/last.pt"
inference_unet_file = "data/last.pt"

# EMA parameters
use_ema = False  # Set to True to use EMA
ema_decay = 0.9999
warmup_steps = 1000

# TEXT TO IMAGE
prompt1 = "A river with boats docked and houses in the background"
prompt2 = "A piece of chocolate swirled cake on a plate"
prompt3 = "A large bed sitting next to a small Christmas Tree surrounded by pictures"
prompt4 = "A bear searching for food near the river"
prompts = [prompt1, prompt2, prompt3, prompt4]
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 3  # min: 1, max: 14
num_samples = 1

# SAMPLER
sampler = "ddpm"
num_inference_steps = 50
seed = 42