from transformers import AutoFeatureExtractor, HubertForSequenceClassification
from transformers import HubertModel
import librosa
import torch

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "facebook/hubert-base-ls960"
)

model = HubertForSequenceClassification.from_pretrained(
    "facebook/hubert-base-ls960",
    num_labels=3
)

base_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

audio, sr = librosa.load("parler_tts_demo/sample_000000.wav", sr=16000)

inputs = feature_extractor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

print("Logits shape: ",(outputs.logits.shape))
print("Logits: ",(outputs.logits))

hidden_state=outputs.hidden_states
print("No. of hidden states: ", (len(hidden_state)))

for i, h in enumerate(hidden_state):
    print()
    print(i)
    print(h.shape)
    print(h)
    print()
