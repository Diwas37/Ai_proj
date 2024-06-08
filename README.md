# Interior-stable-difusion

## Installation and environment setup
```
git clone https://github.com/Trgtuan10/Interior-stable-difusion.git

python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Download checkpoint
```
mkdir checkpoints
cd checkpoints
wget https://civitai.com/api/download/models/128713 -O Interior.safetensors
wget https://civitai.com/api/download/models/123908 -O Exterior.safetensors
wget https://civitai.com/api/download/models/195419 -O Interior_lora.safetensors
```

## Run app
```
cd App_demo
streamlit run streamlit_app
```
