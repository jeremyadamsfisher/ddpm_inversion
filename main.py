import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

import utils

# Primary hyperparameters
ETA = 1 # η
T_SKIP = 36
N_DIFFUSION_STEPS = 100

DEVICE = "mps"
MODEL_ID = "CompVis/stable-diffusion-v1-4"
CFG_SRC = 3.5
CFG_TGT = 15
INITIAL_IMAGE = "horse_mud.jpg"
PROMPT_SRC = "a photo of a horse in the mud"
PROMPT_TGT = "a photo of a horse in the snow"

rng = torch.manual_seed(42)

pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID).to(DEVICE)
pipe.scheduler.set_timesteps(N_DIFFUSION_STEPS)

img = Image.open(INITIAL_IMAGE)
x0 = utils.compress(img, vae=pipe.vae)


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

    cond_embedding = tokenize_and_encode(prompt, pipe.tokenizer.model_max_length)
    _, max_length, _ = cond_embedding.shape
    uncond_embedding = tokenize_and_encode("", max_length)
    return torch.cat([cond_embedding, uncond_embedding])


# Algorithm 1
# TODO: note that timesteps is not guaranteed to be in order, verify this works as expected

N_LATENT_DIMS = pipe.unet.config.sample_size
alpha_bar = pipe.scheduler.alphas_cumprod

xts = torch.zeros(
    (N_DIFFUSION_STEPS, pipe.unet.config.in_channels, N_LATENT_DIMS, N_LATENT_DIMS)
)
for i, t in enumerate(pipe.scheduler.timesteps):
    # See equation 2, this is the crux of the algorithm
    Ɛ̃ = torch.rand_like(x0, device=DEVICE)
    xts[i] = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * Ɛ̃

# Initialize latents
l = torch.randn(
    (1, pipe.unet.config.in_channels, N_LATENT_DIMS, N_LATENT_DIMS), generator=rng
).to(DEVICE)
l *= pipe.scheduler.init_noise_sigma

zs = torch.zeros_like(xts)
for i, t in enumerate(pipe.scheduler.timesteps):
    i_prev = i + 1
    xt = xts[i].unsqueeze(0)
    # Scale the initial noise by the variance required by the scheduler
    latent_model_input = torch.cat([l] * 3)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    chunks = pipe.unet(x0).chunk(2)
    noise_pred_cond, noise_pred_uncond = chunks
    noise_pred = noise_pred_uncond + CFG_SRC * (noise_pred_cond - noise_pred_uncond)
    xt_minus_1 = xts[i_prev].unsqueeze(0)
    x0_pred = (xt - torch.sqrt(1-alpha_bar[t]) * noise_pred)

    # Equation 4
    mu_xt = torch.sqrt(alpha_bar[t+1]) + ...
    variance_t = ...

    zt = (xt_minus_1 - mu_xt) / torch.sqrt(variance_t)