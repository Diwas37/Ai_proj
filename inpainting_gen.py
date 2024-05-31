from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, StableDiffusionPipeline
from utils_func import create_scheduler
from PIL import Image
from diffusers.utils import load_image
import numpy as np
import torch

# from a image, inpaint it and make a new image that contain optinal object in mask 

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

def load_model_inpaint(sd_pipeline: StableDiffusionPipeline):
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11p_sd15_inpaint',
        variant="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")
    
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
    ).to('cuda')
        
    pipe.enable_model_cpu_offload()
    
    return pipe

def inpaint_gen(pipe, 
                init_image, 
                image_mask, 
                prompt, 
                seed=43, 
                num_images=1, 
                num_infer_steps=20, 
                height=512, 
                width=512):
    
    image_control = make_inpaint_condition(init_image, image_mask)
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    images = pipe(
        prompt=prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=num_infer_steps,
        guidance_scale=7,
        image=init_image,
        eta=1.0,
        generator=generator,
        mask_image = image_mask,
        control_image=image_control
    )
    return images.images


if __name__ == "__main__":
    sd_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                          use_safetensors = True,
                                                          torch_dtype=torch.float16).to("cuda")
    pipe = load_model_inpaint(sd_pipeline)
    
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
                         
