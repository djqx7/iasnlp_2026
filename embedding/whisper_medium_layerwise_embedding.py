import os
import numpy as np
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------
input_root = "new_train_audio_batches"
output_root = "hubert_layer_embeddings"

# --------------------------------------------------
# Model
# --------------------------------------------------
model_name = "facebook/hubert-base-ls960"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = HubertModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

num_layers = model.config.num_hidden_layers + 1  # 13

# --------------------------------------------------
# Create layer directories
# --------------------------------------------------
for layer_idx in range(num_layers):
    os.makedirs(
        os.path.join(output_root, f"layer_{layer_idx}"),
        exist_ok=True
    )

# --------------------------------------------------
# Iterate through batches
# --------------------------------------------------
batch_dirs = sorted([
    d for d in os.listdir(input_root)
    if os.path.isdir(os.path.join(input_root, d))
])

for batch_name in batch_dirs:

    batch_path = os.path.join(input_root, batch_name)

    audio_files = [
        f for f in os.listdir(batch_path)
        if f.lower().endswith((".wav", ".flac", ".mp3"))
    ]

    for file_name in tqdm(
        audio_files,
        desc=batch_name
    ):

        audio_path = os.path.join(batch_path, file_name)

        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)

        waveform = waveform.squeeze()

        if sr != 16000:
            waveform = torchaudio.functional.resample(
                waveform,
                sr,
                16000
            )

        inputs = feature_extractor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(
                inputs.input_values.to(device),
                output_hidden_states=True
            )

        hidden_states = outputs.hidden_states

        base_name = os.path.splitext(file_name)[0]

        for layer_idx, hidden in enumerate(hidden_states):

            layer_batch_dir = os.path.join(
                output_root,
                f"layer_{layer_idx}",
                batch_name
            )

            os.makedirs(layer_batch_dir, exist_ok=True)

            save_path = os.path.join(
                layer_batch_dir,
                f"{base_name}.npy"
            )

            np.save(
                save_path,
                hidden.squeeze(0).cpu().numpy()
            )
