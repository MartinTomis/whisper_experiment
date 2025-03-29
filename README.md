## End-to-End pipeline for audio analytics with open-cource models
- whisper-large-v3 - Transkription
- pyannote - Diarisation
- Mistral-7B-Instruct-v0.3 - Small LLM
- Helsinki-NLP/opus-mt-cs-en - translation CS to EN (optional)


## Set up
### 1. Prerequisits
Account on HuggingFace.com is required. Access token should be saved as variable HF_TOKEN in .env.
Add or copy .env into the main dir.


### 2. Installation
mkdir whisper_experiment
cd whisper_experiment
git clone https://github.com/MartinTomis/whisper_experiment/
mkdir models
cd models
mkdir models/Helsinki-NLP/
cd ../
python huggingface_hub_helper.py

### 3. Check CUDA
nvidia-smi

If it fails, CUDA is not installed, but it may give a hint how it can be installed. e.g. with:

Command 'nvidia-smi' not found, but can be installed with:
sudo apt install nvidia-utils-390         # version 390.157-0ubuntu0.22.04.2, or
sudo apt install nvidia-utils-418-server  # version 418.226.00-0ubuntu5~0.22.04.1 or
sudo apt install nvidia-utils-535         # version 535.183.01-0ubuntu0.22.04.1

Then intall it with 
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-utils-535

sudo reboot

nvidia-smi


### 4. Running it
python main.py audio.wav
. If at least drivers are present (), then it can be installed as:


