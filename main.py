import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

import utils

DEVICE = "mps"
MODEL_ID = "CompVis/stable-diffusion-v1-4"
CFG_SRC = 3.5
CFG_TGT = 15
N_DIFFUSION_STEPS = 100
SKIP = 36
INITIAL_IMAGE = "horse_mud.jpg"
PROMPT_SRC = "a photo of a horse in the mud"
PROMPT_TGT = "a photo of a horse in the snow"


pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID).to(DEVICE)
pipe.scheduler.set_timesteps(N_DIFFUSION_STEPS)

x0 = Image.open(INITIAL_IMAGE)
w0 = utils.compress(x0, vae=pipe.vae)

# Compute x_i's and z_i's for i \in {1...T}

def embed_prompt(prompt):
    def tokenize_and_encode(max_length):
        prompt_tokens = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            return pipe.text_encoder(
                prompt_tokens.input_ids.to(DEVICE)
            ).last_hidden_state
    cond_embedding = tokenize_and_encode(
        prompt, pipe.tokenizer.model_max_length
    )
    _, max_length, _ = cond_embedding.shape
    uncond_embedding = tokenize_and_encode("", max_length)
    return torch.cat([cond_embedding, uncond_embedding])

# Algorithm 1, loop 1
u = pipe.unet.config
xts = torch.zeros((
    N_DIFFUSION_STEPS, u.in_channels, u.sample_size, u.sample_size,
))
alpha_bar = pipe.scheduler.alphas_cumprod
for i, t in enumerate(pipe.scheduler.timesteps):
    Ɛ̃ = torch.rand_like(x0)
    xts[i] = (
        torch.sqrt()
    )