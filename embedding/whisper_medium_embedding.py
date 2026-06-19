from transformers import WhisperProcessor, WhisperModel
import librosa
import torch

model_name = "openai/whisper-medium"

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperModel.from_pretrained(model_name)

audio, sr = librosa.load(
    "parler_tts_demo/sample_000000.wav",
    sr=16000
)

inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    encoder_outputs=model.encoder(**inputs, output_hidden_states=True);
    encoder_hidden_layers= encoder_outputs.hidden_states;
    
print("Length of hidden layers: ", (len(encoder_hidden_layers)));

for i, h in enumerate(encoder_hidden_layers):
    print()
    print(i)
    # print(encoder_hidden_layers.shape);
    print(encoder_hidden_layers);
    print()
