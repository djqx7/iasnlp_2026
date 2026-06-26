import os
import numpy as np
from tqdm import tqdm

INPUT_ROOT = "hubert_layer_embeddings"
OUTPUT_ROOT = "mean_pooled_embeddings"

for layer_name in sorted(os.listdir(INPUT_ROOT)):

    layer_path = os.path.join(INPUT_ROOT, layer_name)

    if not os.path.isdir(layer_path):
        continue

    print(f"Processing {layer_name}")

    for root, _, files in os.walk(layer_path):

        rel_path = os.path.relpath(root, layer_path)

        save_dir = os.path.join(
            OUTPUT_ROOT,
            layer_name,
            rel_path
        )

        os.makedirs(save_dir, exist_ok=True)

        for fname in tqdm(files, leave=False):

            if not fname.endswith(".npy"):
                continue

            fpath = os.path.join(root, fname)

            emb = np.load(fpath)

            # emb shape: (T,768)
            pooled = emb.mean(axis=0)

            # pooled shape: (768,)
            np.save(
                os.path.join(save_dir, fname),
                pooled.astype(np.float32)
            )

print("Done.")