from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
import torch
from utils_func import create_scheduler
import cv2
from PIL import Image
import numpy as np

def load_controlnet_model():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
    )
    
    pipe_controlnet = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet, 
        safety_checker=None,
        torch_dtype=torch.float16
    ).to("cuda")
    
    pipe_controlnet.load_lora_weights("TrgTuan10/Interior", weight_name="xsarchitectural-7.safetensors", adapter_name="architecture")
    trigger_words = " ,VERRIERES, DAYLIGHTINDIRECT, LIGHTINGAD, MAGAZINE8K, CINEMATIC LIGHTING, EPIC COMPOSITION"
    
    return pipe_controlnet, trigger_words

def preprocessor_image(image):
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image
    

def gen_interior_controlnet(pipe_controlnet,
                            prompt,
                            trigger_words,
                            neg="",
                            seed=42,
                            num_inference_steps=35,
                            height=512, 
                            width=512,
                            num_images=1,
                            image=None):
    prompt = prompt + " high quality, lightning, luxury" + trigger_words
    scheduler = create_scheduler()
    image_control = preprocessor_image(image)
    image_control.save("control.jpg")
    generator = torch.Generator(device="cuda").manual_seed(seed)

    
    images = pipe_controlnet(
        prompt=prompt, 
        negative_prompt=neg,
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_prompt=num_images,
        image = image_control, 
        height=height, 
        width=width,
        scheduler=scheduler,
    ).images
    
    return images


if __name__ == "__main__":
    pipe_controlnet, trigger_words = load_controlnet_model()
    prompt = "a living room with a red wardrobe, 2 beautiful small trees, red table and 2 sofa, no lighting from windows"
    negative_prompt = "(multiple outlets:1.9),carpets,(multiple tv screens:1.9),2 tables,lamps,lightbuble,(plants:1.6)bad-hands-5, ng_deepnegative_v1_75t, EasyNegative, bad_prompt_version2, bad-artist-anime, bad-artist, bad-image-v2-39000, verybadimagenegative_v1.3, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    image = Image.open("/home/ubuntu/code/trgtuan/Interior-stable-difusion/interior.jpg")
    images = gen_interior_controlnet(pipe_controlnet, prompt, trigger_words, neg=negative_prompt, num_images=1, image=image)
    image = images[0]
    image.save("interior_control.jpg")
    print("Image saved as interior_control.jpg")
    
                            
                            
