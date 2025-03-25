# https://huggingface.co/docs/transformers/main/en/model_doc/whisper

import torch
import torchaudio
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration, pipeline

device = torch.device("cpu")


model_id = "openai/whisper-large-v3"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)

waveform, sample_rate = torchaudio.load("/Users/martintomis/projects/Trask/2367d614-0d52-49d4-a9cd-0ff9dd2dbd72_20240624T06_58_UTC.wav")

# Whisper expects 16kHz mono
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Convert stereo to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)


waveform = waveform.squeeze(0)  # [num_samples]

# Constants
chunk_duration = 30  # seconds
chunk_size = chunk_duration * 16000  # samples per chunk

# Split into chunks
chunks = [waveform[i:i+chunk_size] for i in range(0, waveform.shape[0], chunk_size)]

# Transcribe each chunk
transcripts = []

for idx, chunk in enumerate(chunks):
    inputs = processor(chunk.numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    generated_ids = model.generate(
        input_features,
        max_length=448,
        num_beams=1,
        do_sample=False
    )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    transcripts.append(text)
    print(f"Chunk {idx+1} done.")

# Combine results
full_transcription = " ".join(transcripts)
print("\nFull transcription:\n", full_transcription)