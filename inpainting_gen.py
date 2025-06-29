from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, StableDiffusionPipeline
from utils_func import create_scheduler
from PIL import Image
from diffusers.utils import load_image
import numpy as np
import torch

# from a image, inpaint it and make a new image that contain optinal object in mask 

def make_inpaint_condition(image, image_mask):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[:2] == image_mask.shape[:2], "image and image_mask must have the same image size"
    
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).float()  # converting to float is default, specify if needed
    image = image.to(device, dtype=torch.float16 if device=="cuda" else torch.float32)  # move to GPU as float16

    image_mask = torch.from_numpy(np.expand_dims(image_mask, 0)).to(device, dtype=torch.float16 if device=="cuda" else torch.float32)

    return image, image_mask

def load_model_inpaint(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11p_sd15_inpaint',
        variant="fp16" if device=="cuda" else None,
        torch_dtype=torch.float16 if device=="cuda" else None,
    ).to(device)
    
    if device == "cuda":
        sd_pipeline = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16).to(device)
    else:
        sd_pipeline = StableDiffusionPipeline.from_single_file(model_path).to(device)
    
    pipe = StableDiffusionControlNetInpaintPipeline(
        vae=sd_pipeline.vae,
        text_encoder=sd_pipeline.text_encoder,
        tokenizer=sd_pipeline.tokenizer,
        unet=sd_pipeline.unet,
        scheduler=sd_pipeline.scheduler,
        requires_safety_checker=False,
        safety_checker=None,
        feature_extractor=sd_pipeline.feature_extractor,
        controlnet=controlnet,
    ).to(device)
    
    return pipe

def inpaint_gen(pipe, 
                init_image, 
                image_mask, 
                prompt,
                neg="", 
                seed=43, 
                num_images=1, 
                num_infer_steps=35, 
                height=512, 
                width=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    neg += " soft line, curved line, sketch, ugly, logo, pixelated, lowres, text, word, cropped, low quality, normal quality, username, watermark, signature, blurry, soft"

    image_control, image_mask_tensor = make_inpaint_condition(init_image, image_mask)
    
    init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).unsqueeze(0)
    init_image = init_image.to(device)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    images = pipe(
        prompt=prompt,
        negative_prompt=neg,
        num_images_per_prompt=num_images,
        num_inference_steps=num_infer_steps,
        guidance_scale=7,
        image=init_image,
        eta=1.0,
        generator=generator,
        mask_image=image_mask_tensor,
        control_image=image_control,
    )
    return images.images

if __name__ == "__main__":
    pipe = load_model_inpaint("checkpoints/Interior.pt")
    
    init_image = load_image(
    "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png"
    )
    init_image = init_image.resize((512, 512))
    mask_image = load_image(
    "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png"
    )
    mask_image = mask_image.resize((512, 512))
    
    images = inpaint_gen(pipe = pipe,
                        init_image = init_image,
                        image_mask = mask_image,
                        prompt = "a handsome man with ray-ban sunglasses",
                        num_images=1)
    images[0].save("inpaint.jpg")

