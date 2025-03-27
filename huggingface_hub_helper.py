from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


snapshot_download(
    repo_id="pyannote/speaker-diarization",
    local_dir="models/pyannote_model",
    use_auth_token=hf_token
)

snapshot_download(
    repo_id="openai/whisper-large-v3",
    local_dir="models/whisper-large-v3",
    local_dir_use_symlinks=False  # optional: copies instead of symlinks
)