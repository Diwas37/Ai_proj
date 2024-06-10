<div align="center">
<h1> Interior-stable-difusion: Revolutionizing Interior Design with Rapid Visualization and Customization
</div>
The evolution of AI technologies like [Stable Diffusion](https://arxiv.org/abs/2112.10752) has revolutionized visual design. Now, with "Interior-Stable-Diffusion," this technology is tailored for interior design, enabling rapid generation, style modification, and object replacement in interior spaces. This application empowers designers to visualize and refine spaces with unprecedented speed and precision, transforming ideas into reality in moments.

## Main Functions
Using Stable Diffusion, this app can make desirable images with three main function: 
- General generation: Utilize the [text2img pipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img) o create detailed interior images from textual descriptions.
- Fixing style: Utilize the [Controlnet Canny](https://huggingface.co/lllyasviel/sd-controlnet-canny) echnique to maintain the original image’s edges while introducing a new style based on the canny edge map.
- Replacing object: using [ControlnetInpaintPipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetInpaintPipeline) to seamlessly replace objects within specified masked areas of an image.
## Examples

## Installation and Usage
### Environment setup
```
python3 -m venv .env
source .env/bin/activate

git clone https://github.com/Trgtuan10/Interior-stable-difusion.git
cd Interior-stable-difusion
pip install -r requirements.txt
```

### Download my checkpoint
```
mkdir checkpoints
cd checkpoints
wget https://civitai.com/api/download/models/128713 -O Interior.safetensors
wget https://civitai.com/api/download/models/195419 -O Interior_lora.safetensors
```

### Run app
```
cd App_demo
streamlit run streamlit_app
```
