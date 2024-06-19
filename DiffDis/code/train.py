import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from ddpm import DDPMSampler
from pipeline import get_time_embedding
from dataloader import train_dataloader
import model_loader
import time
from config import *
from diffusion import TransformerBlock, UNet_Transformer  # Ensure these are correctly imported

import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer

# Load the models
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)
ddpm = DDPMSampler(generator=None)

if train_unet_file is not None:
    # Load the UNet model
    print(f"Loading UNet model from {train_unet_file}")
    models['diffusion'].load_state_dict(torch.load(train_unet_file)['model_state_dict'])
    if 'best_loss' in torch.load(train_unet_file):
        best_loss = torch.load(train_unet_file)['best_loss']
        best_step = torch.load(train_unet_file)['best_step']
        last_loss = torch.load(train_unet_file)['last_loss']
        last_step = torch.load(train_unet_file)['last_step']
    else:
        best_loss = float('inf')
        best_step = 0
        last_loss = 0.0
        last_step = 0
else:
    best_loss = float('inf')
    best_step = 0
    last_loss = 0.0
    last_step = 0

# TEXT TO IMAGE
tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")

# Disable gradient computations for the models['encoder'], DDPM, and models['clip'] models
for param in models['encoder'].parameters():
    param.requires_grad = False

for param in models['clip'].parameters():
    param.requires_grad = False

# Set the models['encoder'] and models['clip'] to eval mode
models['encoder'].eval()
models['clip'].eval()

# Separate parameters for discriminative tasks
discriminative_params = []
non_discriminative_params = []

for name, param in models['diffusion'].named_parameters():
    if isinstance(getattr(models['diffusion'], name.split('.')[0], None), (TransformerBlock, UNet_Transformer)):
        discriminative_params.append(param)
    else:
        non_discriminative_params.append(param)

# AdamW optimizer with separate learning rates
optimizer = torch.optim.AdamW([
    {'params': non_discriminative_params, 'lr': learning_rate},
    {'params': discriminative_params, 'lr': discriminative_learning_rate}
], betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay, eps=adam_epsilon)

if train_unet_file is not None:
    print(f"Loading optimizer state from {train_unet_file}")
    optimizer.load_state_dict(torch.load(train_unet_file)['optimizer_state_dict'])

# Linear warmup scheduler for non-discriminative parameters
def warmup_lr_lambda(current_step: int):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[
    warmup_lr_lambda,  # Apply warmup for non-discriminative params
    lambda step: 1.0  # Keep constant learning rate for discriminative params
])

# EMA setup
if use_ema:
    ema_unet = torch.optim.swa_utils.AveragedModel(models['diffusion'], avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: ema_decay * averaged_model_parameter + (1 - ema_decay) * model_parameter)


def train(num_train_epochs, device="cuda", save_steps=1000):
    global best_loss, best_step, last_loss, last_step

    if train_unet_file is not None:
        first_epoch = last_step // len(train_dataloader)
        global_step = last_step + 1
    else:
        first_epoch = 0
        global_step = 0

    accumulator = 0

    # Move models to the device
    models['encoder'].to(device)
    models['clip'].to(device)
    models['diffusion'].to(device)
    if use_ema:
        ema_unet.to(device)

    num_train_epochs = tqdm(range(first_epoch, num_train_epochs), desc="Epoch")
    for epoch in num_train_epochs:
        train_loss = 0.0
        num_train_steps = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()

            # Extract images and texts from batch
            images = batch["pixel_values"]
            texts = batch["input_ids"]

            # Move batch to the device
            images = images.to(device)
            texts = texts.to(device)

            # Encode images to latent space
            encoder_noise = torch.randn(images.shape[0], 4, LATENTS_HEIGHT, LATENTS_WIDTH).to(device)  # Shape (BATCH_SIZE, 4, 32, 32)
            latents = models['encoder'](images, encoder_noise)

            # Sample noise and timesteps for diffusion process
            bsz = latents.shape[0]
            timesteps = torch.randint(0, ddpm.num_train_timesteps, (bsz,), device=latents.device).long()
            text_timesteps = torch.randint(0, ddpm.num_train_timesteps, (bsz,), device=latents.device).long()

            # Add noise to latents and texts
            noisy_latents, image_noise = ddpm.add_noise(latents, timesteps)
            encoder_hidden_states = models['clip'](texts)
            noisy_text_query, text_noise = ddpm.add_noise(encoder_hidden_states, text_timesteps)

            # Get time embeddings
            image_time_embeddings = get_time_embedding(timesteps, is_image=True).to(device)
            text_time_embeddings = get_time_embedding(timesteps, is_image=False).to(device)
            
            # Average and normalize text time embeddings
            average_noisy_text_query = noisy_text_query.mean(dim=1)
            text_query = F.normalize(average_noisy_text_query, p=2, dim=-1)

            # Randomly drop 10% of text and image conditions: Context Free Guidance
            if torch.rand(1).item() < 0.1:
                text_query = torch.zeros_like(text_query)
            if torch.rand(1).item() < 0.1:
                noisy_latents = torch.zeros_like(noisy_latents)

            # Predict the noise residual and compute loss
            image_pred, text_pred = models['diffusion'](noisy_latents, encoder_hidden_states, image_time_embeddings, text_time_embeddings, text_query)
            image_loss = F.mse_loss(image_pred.float(), image_noise.float(), reduction="mean")
            
            if use_contrastive_loss:
                # Assuming margin is predefined, e.g., margin = 1.0
                margin = 1.0
                # Assuming all pairs are similar
                target = torch.ones(text_pred.size(0)).to(device)
                text_loss = F.cosine_embedding_loss(text_pred.float(), text_query.float(), target, margin)
            else:
                text_loss = F.mse_loss(text_pred.float(), text_query.float(), reduction="mean")
            
            loss = image_loss + Lambda * text_loss
            train_loss += loss.item()
            accumulator += loss.item()

            # Backpropagate
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if use_ema:
                ema_unet.update_parameters(models['diffusion'])

            end_time = time.time()

            if train_unet_file is not None and epoch == first_epoch:
                print(f"Step: {step+1+last_step}/{num_train_steps+last_step}   Loss: {loss.item()}   Time: {end_time - start_time}", end="\r")
            else:
                print(f"Step: {step}/{num_train_steps}   Loss: {loss.item()}   Time: {end_time - start_time}", end="\r")

            if global_step % save_steps == 0 and global_step > 0:
                # Check if the current step's loss is the best
                if accumulator / save_steps < best_loss:
                    best_loss = accumulator / save_steps
                    best_step = global_step
                    best_save_path = os.path.join(train_output_dir, "best.pt")
                    if use_ema:
                        torch.save({
                            'model_state_dict': models['diffusion'].state_dict(),
                            'ema_state_dict': ema_unet.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_loss': best_loss,
                            'best_step': best_step,
                            'last_loss': accumulator / save_steps,
                            'last_step': global_step
                        }, best_save_path) 
                    else:
                        torch.save({
                            'model_state_dict': models['diffusion'].state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_loss': best_loss,
                            'best_step': best_step,
                            'last_loss': accumulator / save_steps,
                            'last_step': global_step
                        }, best_save_path)              

                    print(f"\nNew best model saved to {best_save_path} with loss {best_loss}")

                # Save model and optimizer state
                last_save_path = os.path.join(train_output_dir, f"last.pt")
                if use_ema:
                    torch.save({
                        'model_state_dict': models['diffusion'].state_dict(),
                        'ema_state_dict': ema_unet.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss,
                        'best_step': best_step,
                        'last_loss': accumulator / save_steps,
                        'last_step': global_step
                    }, last_save_path)
                else:
                    torch.save({
                        'model_state_dict': models['diffusion'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss,
                        'best_step': best_step,
                        'last_loss': accumulator / save_steps,
                        'last_step': global_step
                    }, last_save_path)
                    
                print(f"Saved state to {last_save_path}")

                # Generate samples from the model
                for i, prompt in enumerate(prompts):
                    # Sample images from the model
                    output_image = pipeline.generate(
                        prompt=prompt,
                        uncond_prompt=uncond_prompt,
                        input_image=None,
                        strength=0.9,
                        do_cfg=do_cfg,
                        cfg_scale=cfg_scale,
                        sampler_name=sampler,
                        n_inference_steps=num_inference_steps,
                        seed=seed,
                        models=models,
                        device=DEVICE,
                        idle_device=DEVICE,
                        tokenizer=tokenizer,
                    )

                    # Save the generated image
                    output_image = Image.fromarray(output_image)
                    output_image.save(os.path.join(train_output_dir, "images", "prompt" + str(i+1), f"step{global_step}.png"))
                
                print(f"\nSaved images for step {global_step}")
                s = 'Epoch: %d   Step: %d   Loss: %.5f   Best Loss: %.5f   Best Step: %d\n' % (epoch+1, global_step, accumulator / save_steps, best_loss, best_step)
                print(s)
                with open(os.path.join(train_output_dir, 'train_log.txt'), 'a') as f:
                    f.write(s)

                accumulator = 0.0

            global_step += 1

        print(f"Average loss over epoch: {train_loss / (step + 1)}")


if __name__ == "__main__":
    s = '==> Training starts..'
    s += f'\n\nModel file: {model_file}'
    s += f'\nUNet file: {train_unet_file}'
    s += f'\nBatch size: {BATCH_SIZE}'
    s += f'\nWidth: {WIDTH}'
    s += f'\nHeight: {HEIGHT}'
    s += f'\nLatents width: {LATENTS_WIDTH}'
    s += f'\nLatents height: {LATENTS_HEIGHT}'
    s += f'\nFirst epoch: {last_step // len(train_dataloader)}'
    s += f'\nNumber of training epochs: {num_train_epochs}'
    s += f'\nLambda: {Lambda}'
    s += f'\nLearning rate: {learning_rate}'
    s += f'\nDiscriminative learning rate: {discriminative_learning_rate}'
    s += f'\nAdam beta1: {adam_beta1}'
    s += f'\nAdam beta2: {adam_beta2}'
    s += f'\nAdam weight decay: {adam_weight_decay}'
    s += f'\nAdam epsilon: {adam_epsilon}'
    s += f'\nUse contrastive loss: {use_contrastive_loss}'
    s += f'\nUse EMA: {use_ema}'
    s += f'\nEMA decay: {ema_decay}'
    s += f'\nWarmup steps: {warmup_steps}'
    s += f'\nOutput directory: {train_output_dir}'
    s += f'\nSave steps: {save_steps}'
    s += f'\nDevice: {DEVICE}'
    s += f'\nSampler: {sampler}'
    s += f'\nNumber of inference steps: {num_inference_steps}'
    s += f'\nSeed: {seed}'
    for i, prompt in enumerate(prompts):
        s += f'\nPrompt {i + 1}: {prompt}'
    s += f'\nUnconditional prompt: {uncond_prompt}'
    s += f'\nDo CFG: {do_cfg}'
    s += f'\nCFG scale: {cfg_scale}'
    s += f'\n\n'
    print(s)

    # Create the output directory
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(os.path.join(train_output_dir, "images"), exist_ok=True)
    for i in range(len(prompts)):
        os.makedirs(os.path.join(train_output_dir, "images", "prompt" + str(i+1)), exist_ok=True)

    with open(os.path.join(train_output_dir, 'train_log.txt'), 'w') as f:
        f.write(s)

    train(num_train_epochs=num_train_epochs, device=DEVICE, save_steps=save_steps)
