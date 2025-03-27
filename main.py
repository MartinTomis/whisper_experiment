# https://huggingface.co/docs/transformers/main/en/model_doc/whisper


import os
import sys
from dotenv import load_dotenv
import torch
torch.set_num_threads(4)
import torchaudio
from datasets import load_dataset
from transformers import AutoProcessor,AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoModelForCausalLM, WhisperForConditionalGeneration, pipeline,MBartForConditionalGeneration, MBart50TokenizerFast

from pyannote.audio import Pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if len(sys.argv) < 2:
    print("Usage: python your_script.py <audio_file.wav>")
    sys.exit(1)


if torch.cuda.is_available():
    print("✅ GPU is available!")
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("❌ No GPU available.")


load_dotenv()
hf_token = os.getenv("HF_TOKEN")







# Definitions
model_id = "whisper-large-v3" # medium, tiny, base, small, whisper-large-v3
audio_path = sys.argv[1]
print(f"File name is:  {audio_path}")

target_sample_rate = 16000
language = "cs"               
task = "transcribe"


# Transcription model load
#processor = AutoProcessor.from_pretrained(model_id)
print("*******IAutoProcessor*******")
processor = AutoProcessor.from_pretrained('models/'+model_id)
#model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
print("*******AutoModelForSpeechSeq2Seq*******")
model = AutoModelForSpeechSeq2Seq.from_pretrained('models/'+model_id).to(device)
chunk_duration = 30  # seconds
chunk_size = chunk_duration * target_sample_rate  # samples per chunk

# Diarization pipeline load
print("*******INITIALIZING DIARIZATION MODEL*******")
pipeline = Pipeline.from_pretrained("models/pyannote_model/config.yaml", use_auth_token=hf_token)
#pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)





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
print("*******RUNNING DIARIZATION*******")
diarization = pipeline(audio_path )
print("*******DIARIZATION DONE*******")

def format_timestamp(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02}:{s:02}"

segments = []

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn} - {speaker}")
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
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generated_ids = model.generate(
        input_features,
        attention_mask=attention_mask,
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




######
# Prompt example
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained('models/'+model_id)
model = AutoModelForCausalLM.from_pretrained('models/'+model_id, torch_dtype=torch.float16).to(device)

# Example English transcript
transcript = """
Customer: Hi, I’d like to order three boxes of green tea and a packet of honey.
Agent: Sure, would you like anything else?
Customer: No, that will be all for now.
"""

# Create a prompt to ask a question about it
prompt = f"""<s>[INST] What did the customer want to buy? [/INST] {transcript}"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("RESPONDING TO A QUESTION ABOUT THE INTERACTION")
print(response)
####


#################3
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-cs"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Example English sentence
text = "The customer ordered two boxes of green tea and one jar of honey."

# Tokenize and translate
inputs = tokenizer(response, return_tensors="pt", padding=True, truncation=True)
translated = model.generate(**inputs)

# Decode the result
output = tokenizer.decode(translated[0], skip_special_tokens=True)
print(output)

###################