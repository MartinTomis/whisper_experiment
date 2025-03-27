from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


snapshot_download(
    repo_id="Helsinki-NLP/opus-mt-cs-en",
    local_dir="models/Helsinki-NLP/opus-mt-cs-en",
    local_dir_use_symlinks=False  # optional: copies instead of symlinks
)
