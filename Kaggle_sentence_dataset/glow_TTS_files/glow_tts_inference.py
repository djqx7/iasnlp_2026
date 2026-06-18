from TTS.api import TTS
import torch
import os

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# LOAD MODEL (ONLY ONCE)
# --------------------------------------------------

tts = TTS(
    model_name="tts_models/en/ljspeech/glow-tts"
).to(device)

# --------------------------------------------------
# INPUT / OUTPUT
# --------------------------------------------------

PROMPT_FILE = "prompts.txt"
OUTPUT_DIR = "generated_wavs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# READ PROMPTS
# --------------------------------------------------

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompts = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Found {len(prompts)} prompts.")

# --------------------------------------------------
# GENERATE AUDIO
# --------------------------------------------------

for idx, text in enumerate(prompts, start=1):

    output_path = os.path.join(
        OUTPUT_DIR,
        f"sample_{idx:04d}.wav"
    )

    print(f"[{idx}/{len(prompts)}] Generating: {output_path}")

    tts.tts_to_file(
        text=text,
        file_path=output_path
    )

print("\nDone!")
print(f"Generated {len(prompts)} wav files.")
