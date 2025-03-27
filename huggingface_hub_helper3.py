from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir="models/Mistral-7B-Instruct-v0.3",
    local_dir_use_symlinks=False  # optional: copies instead of symlinks
)
