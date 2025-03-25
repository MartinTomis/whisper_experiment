# https://huggingface.co/docs/transformers/main/en/model_doc/whisper

import torch
import torchaudio
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration, pipeline

device = torch.device("cpu")

# Definitions
model_id = "openai/whisper-large-v3" # medium, tiny, base, small
target_sample_rate = 16000
language = "cs"               
task = "transcribe"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
chunk_duration = 30  # seconds
chunk_size = chunk_duration * target_sample_rate  # samples per chunk


# When not using Pipeline, process audio with torchaudio
waveform, sample_rate = torchaudio.load("/Users/martintomis/projects/Trask/2367d614-0d52-49d4-a9cd-0ff9dd2dbd72_20240624T06_58_UTC.wav")
num_channels, total_samples = waveform.shape

assert num_channels == 2, "This pipeline expects stereo (2-channel) audio."

# Resample to 16kHz if needed
if sample_rate != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
    waveform = resampler(waveform)



# Chunking
def split_chunks(channel_audio, chunk_sec=30):
    chunk_size = chunk_sec * target_sample_rate
    return [channel_audio[i:i+chunk_size] for i in range(0, channel_audio.shape[0], chunk_size)]

# Channel 0 → Speaker 1, Channel 1 → Speaker 2
speaker_chunks = {
    "Speaker 1": split_chunks(waveform[0], chunk_duration),
    "Speaker 2": split_chunks(waveform[1], chunk_duration),
}


def transcribe(audio_tensor, speaker_label, index):
    inputs = processor(
        audio_tensor.numpy(),
        sampling_rate=target_sample_rate,
        return_tensors="pt",
        language=language,
        task=task,
    )
    input_features = inputs.input_features.to(device)

    generated_ids = model.generate(
        input_features,
        max_length=448,
        num_beams=1,
        do_sample=False
    )
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return f"[{speaker_label} - Chunk {index+1}] {text}"

results = []

for speaker, chunks in speaker_chunks.items():
    for idx, chunk in enumerate(chunks):
        if chunk.shape[0] < 1000:
            continue  # skip super short chunks
        print(f"Transcribing {speaker} - chunk {idx+1}/{len(chunks)}...")
        result = transcribe(chunk, speaker, idx)
        results.append(result)

# --- OUTPUT ---
final_transcription = "\n".join(results)
print("\n--- Final Transcription ---\n")
print(final_transcription)