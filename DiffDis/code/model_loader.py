from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter

def load_state_dict_ignore_size_mismatch(model, state_dict):
    for name, param in model.named_parameters():
        if name in state_dict and param.size() == state_dict[name].size():
            param.data = state_dict[name].data

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=False)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=False)

    diffusion = Diffusion().to(device)
    # diffusion.load_state_dict(state_dict['diffusion'], strict=False)
    load_state_dict_ignore_size_mismatch(diffusion, state_dict['diffusion'])

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=False)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }