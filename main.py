# https://huggingface.co/docs/transformers/main/en/model_doc/whisper


import os
from dotenv import load_dotenv
import torch
import torchaudio
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration, pipeline
from pyannote.audio import Pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


load_dotenv()
hf_token = os.getenv("HF_TOKEN")



# Definitions
model_id = "openai/whisper-tiny" # medium, tiny, base, small, whisper-large-v3
audio_path = "/Users/martintomis/projects/Trask/2367d614-0d52-49d4-a9cd-0ff9dd2dbd72_20240624T06_58_UTC.wav"
target_sample_rate = 16000
language = "cs"               
task = "transcribe"


# Transcription model load
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
chunk_duration = 30  # seconds
chunk_size = chunk_duration * target_sample_rate  # samples per chunk

# Diarization pipeline load
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)





# When not using Pipeline, process audio with torchaudio
waveform, sample_rate = torchaudio.load(audio_path)
num_channels, total_samples = waveform.shape

# Convert to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Resample to 16kHz if needed
if sample_rate != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
    waveform = resampler(waveform)

waveform = waveform.squeeze(0)  # shape: [num_samples]

# Diarize your audio file (should be mono, 16kHz)
diarization = pipeline(audio_path )

def format_timestamp(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02}:{s:02}"

segments = []

for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_sample = int(turn.start * target_sample_rate )
    end_sample = int(turn.end * target_sample_rate )
    chunk = waveform[start_sample:end_sample]

    inputs = processor(
        chunk.numpy(),
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
    segments.append({
        "speaker": speaker,
        "start": turn.start,
        "end": turn.end,
        "text": text.strip()
    })




for seg in segments:
    start = format_timestamp(seg["start"])
    end = format_timestamp(seg["end"])
    print(f"[{seg['speaker']} | {start} - {end}] {seg['text']}")