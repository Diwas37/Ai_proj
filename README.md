<div align="center">
<h1> Interior-stable-difusion: Revolutionizing Interior Design with Rapid Visualization and Customization
</div>

<div style="display: flex; justify-content: center; flex-wrap: nowrap; overflow-x: auto;">
  <img src="assets/gr1.png" style="flex: 0 0 auto; width: 32%;">
  <img src="assets/inp1.png" style="flex: 0 0 auto; width: 32%;">
  <img src="assets/con3.png" style="flex: 0 0 auto; width: 32%;">
</div>

The evolution of AI technologies like Stable Diffusion(https://arxiv.org/abs/2112.10752) has revolutionized visual design. Now, with "Interior-Stable-Diffusion," this technology is tailored for interior design, enabling rapid generation, style modification, and object replacement in interior spaces. This application empowers designers to visualize and refine spaces with unprecedented speed and precision, transforming ideas into reality in moments.

## Main Functions
Using Stable Diffusion, this app can make desirable images with three main function: 
- General generation: Utilize the [text2img pipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img) to create detailed interior images from textual descriptions.
- Fixing style: Utilize the [Controlnet Canny](https://huggingface.co/lllyasviel/sd-controlnet-canny) echnique to maintain the original image’s edges while introducing a new style based on the canny edge map.
- Replacing object: using [ControlnetInpaintPipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/controlnet#diffusers.StableDiffusionControlNetInpaintPipeline) to seamlessly replace objects within specified masked areas of an image.
## Examples
### General generation
<div style="display: flex; justify-content: center; flex-wrap: nowrap; overflow-x: auto;">
  <img src="assets/prompt1.png" style="flex: 0 0 auto; width: 48%;">
  <img src="assets/prompt2.png" style="flex: 0 0 auto; width: 48%;">
</div>
<p align="center">Prompt: A living room with a TV, wooden floor, a sofa, a nice glass table and a flower in the table</p>
<div style="display: flex; justify-content: center; flex-wrap: nowrap; overflow-x: auto;">
  <img src="assets/prompt3.png" style="flex: 0 0 auto; width: 48%;">
  <img src="assets/prompt4.png" style="flex: 0 0 auto; width: 48%;">
</div>
<p align="center">Prompt: A large modern kitchen with light grey, brown and white, large kitchen cabinets</p>


### Fixing style
<div style="display: flex; justify-content: center; flex-wrap: nowrap; overflow-x: auto;">
  <img src="assets/con1.png" style="flex: 0 0 auto; width: 48%;">
  <img src="assets/con2.png" style="flex: 0 0 auto; width: 48%;">
</div>
<p align="center">Change: A black table</p>
<div style="display: flex; justify-content: center; flex-wrap: nowrap; overflow-x: auto;">
  <img src="assets/con3.png" style="flex: 0 0 auto; width: 48%;">
  <img src="assets/con4.png" style="flex: 0 0 auto; width: 48%;">
</div>
<p align="center">Change: A colorful violet chandelier, darker ceiling.</p>

### Replacing object
<div style="display: flex; justify-content: center; flex-wrap: nowrap; overflow-x: auto;">
  <img src="assets/inp1.png" style="flex: 0 0 auto; width: 32%;">
  <img src="assets/inp2.png" style="flex: 0 0 auto; width: 32%;">
  <img src="assets/inp3.png" style="flex: 0 0 auto; width: 32%;">
</div>
<p align="center">Prompt: a luxury liquor cabinet</p>
<div style="display: flex; justify-content: center; flex-wrap: nowrap; overflow-x: auto;">
  <img src="assets/inp4.png" style="flex: 0 0 auto; width: 32%;">
  <img src="assets/inp5.png" style="flex: 0 0 auto; width: 32%;">
  <img src="assets/inp6.png" style="flex: 0 0 auto; width: 32%;">
</div>
<p align="center">Prompt: a fridge</p>

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
