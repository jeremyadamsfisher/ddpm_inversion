import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

import utils

# Primary hyperparameters
η = 1
T_SKIP = 36
N_DIFFUSION_STEPS = 100

DEVICE = "mps"
MODEL_ID = "CompVis/stable-diffusion-v1-4"
CFG_SCALE_SRC = 3.5
CFG_SCALE_TGT = 15
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
LATENT_SPACE = (
    N_DIFFUSION_STEPS,
    pipe.unet.config.in_channels,
    N_LATENT_DIMS,
    N_LATENT_DIMS,
)
ᾱs = pipe.scheduler.alphas_cumprod
ts = pipe.scheduler.timesteps

xts = torch.zeros(LATENT_SPACE)
for i, t in enumerate(ts):
    # See equation 2, this is the crux of the algorithm
    Ɛ̃ = torch.rand_like(x0, device=DEVICE)
    xts[i] = torch.sqrt(ᾱs[t]) * x0 + torch.sqrt(1 - ᾱs[t]) * Ɛ̃

zs = torch.zeros(LATENT_SPACE)
embedding = embed_prompt(PROMPT_SRC)
for i, t in enumerate(ts):
    i_prev = i + 1
    t_prev = ts[i_prev]

    xt = 
    noise_preds = pipe.unet(x0, t, embedding)
    noise_pred_cond, noise_pred_uncond = noise_preds.chunk(2)

    # Classifier-free guidance
    diff = noise_pred_cond - noise_pred_uncond
    noise_pred = noise_pred_uncond + CFG_SCALE_SRC * diff
    xt_minus_1 = xts[i_prev].unsqueeze(0)
    x0_pred = xts[i].unsqueeze(0) - torch.sqrt(1 - ᾱs[t]) * noise_pred

    # Equation 4
    mu_xt = torch.sqrt(ᾱs[t + 1]) + ...
    variance_t = ...

    zt = (xt_minus_1 - mu_xt) / torch.sqrt(variance_t)
