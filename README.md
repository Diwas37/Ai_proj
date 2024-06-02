# Interior-stable-difusion

## Installation and environment setup
```
git clone 
git clone https://github.com/Trgtuan10/Interior-stable-difusion.git

python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Download checkpoint
```
cd checkpoints
wget https://civitai.com/api/download/models/50722 -O Interior.pt
wget https://civitai.com/api/download/models/123908 -O Exterior.safetensor
```

## Run app
```
cd App_demo
streamlit run streamlit_app
```
