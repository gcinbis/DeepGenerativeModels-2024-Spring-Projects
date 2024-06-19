import torch
import torch.nn.functional as F
import os
import model_loader
import time
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
from config import *

# Load the models
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

if inference_unet_file is not None:
    # Load the UNet model
    print(f"Loading UNet model from {inference_unet_file}")
    models['diffusion'].load_state_dict(torch.load(inference_unet_file)['model_state_dict'])

# TEXT TO IMAGE
tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")

# Create the output directory
os.makedirs(inference_output_dir, exist_ok=True)
for i in range(len(prompts)):
    os.makedirs(os.path.join(inference_output_dir, "prompt" + str(i+1)), exist_ok=True)

# Generate samples from the model
for i, prompt in enumerate(prompts):
    for j in range(num_samples):
        start = time.time()

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

        end = time.time()
        
        print(f"PROMPT {i+1} - SAMPLE {j+1} - TIME: {end - start:.2f}s\n")

        # Save the generated image
        output_image = Image.fromarray(output_image)
        output_image.save(os.path.join(inference_output_dir, "prompt" + str(i+1), f"sample{j+1}.png"))