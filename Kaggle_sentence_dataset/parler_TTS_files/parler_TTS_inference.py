import os
import torch
import pandas as pd
import soundfile as sf

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# ----------------------------
# Configuration
# ----------------------------
CSV_FILE = "parler_tts_demo.csv"
OUTPUT_DIR = "parler_tts_demo"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load model
# ----------------------------
model_name = "parler-tts/parler-tts-mini-v1"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ----------------------------
# Label -> style description
# ----------------------------
def get_description(label):

    label = str(label).lower()

    if label == "question":
        return (
            "A curious speaker asks a question naturally with rising "
            "intonation at the end."
        )

    elif label == "command":
        return (
            "A speaker issues a direct command. The speech is firm, authoritative, and instructional." 
            "The voice is higher energy, slightly louder, and clearly emphasizes key words."
            "The sentence has a strong falling intonation at the end with no softness or hesitation."
        )

    elif label == "statement":
        return (
            "A neutral speaker makes a factual statement. The tone is calm, steady, and conversational."
            "There is no emotional emphasis, and the speech follows a natural rhythm "                   
            "with a gentle falling intonation at the end."
        )

    return "A speaker talks naturally."


# ----------------------------
# Optional punctuation recovery
# ----------------------------
def prepare_text(text, label):

    text = str(text).strip()

    if label == "question" and not text.endswith("?"):
        text += "?"

    elif label == "command" and not text.endswith("!"):
        text += "!"

    elif label == "statement" and not text.endswith("."):
        text += "."

    return text


# ----------------------------
# Read CSV
# ----------------------------
df = pd.read_csv(CSV_FILE)

# ----------------------------
# Generate audio
# ----------------------------
for idx, row in df.iterrows():

    text = row["text"]
    label = row["label"]

    text = prepare_text(text, label)
    description = get_description(label)

    input_ids = tokenizer(
        description,
        return_tensors="pt"
    ).input_ids.to(device)

    prompt_ids = tokenizer(
        text,
        return_tensors="pt"
    ).input_ids.to(device)

    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_ids
        )

    audio = generation.cpu().numpy().squeeze()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"sample_{idx:06d}.wav"
    )

    sf.write(
        output_path,
        audio,
        model.config.sampling_rate
    )

    print(f"Saved: {output_path}")

print("Done.")
