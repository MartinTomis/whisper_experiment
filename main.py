# https://huggingface.co/docs/transformers/main/en/model_doc/whisper

import torch
from datasets import load_dataset
from transformers import AutoProcessor, WhisperForConditionalGeneration, pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-medium", device=-1)
result = pipe("/Users/martintomis/projects/Trask/2367d614-0d52-49d4-a9cd-0ff9dd2dbd72_20240624T06_58_UTC.wav",return_timestamps=True)
print(result["text"])