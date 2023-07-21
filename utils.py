import torch
from PIL import Image
from torchvision import transforms

VAE_TO_UNET_SCALING_FACTOR = 0.18215


to_tensor = transforms.ToTensor()


def compress(
    img: Image.Image,  # Input image
    vae: torch.nn.Module,  # VAE
):
    """Project pixels into latent space"""
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    img = to_tensor(img).unsqueeze(0).to(vae.device)
    img = img * 2 - 1  # Note scaling
    with torch.no_grad():
        latents = vae.encode(img)
    return VAE_TO_UNET_SCALING_FACTOR * latents.latent_dist.sample()


def decompress(
    latents: torch.Tensor,  # VAE latents
    vae: torch.nn.Module,  # VAE
    as_pil=True,  # Return a PIL image
    no_grad=True,  # Discard forward gradientss
):
    """Project latents into pixel space"""
    if no_grad:
        with torch.no_grad():
            img = vae.decode(latents / VAE_TO_UNET_SCALING_FACTOR).sample
    else:
        img = vae.decode(latents / VAE_TO_UNET_SCALING_FACTOR).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    # color dimension goes last for matplotlib
    img = img.permute(0, 2, 3, 1)
    if as_pil:
        img = img.cpu().numpy().squeeze()
        img = (img * 255).round().astype("uint8")
        img = Image.fromarray(img)
    return img
