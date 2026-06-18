import json
import pandas as pd

with open("metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for dialog_id, utterances in data.items():
    for utterance_id, details in utterances.items():

        wav_file = (
            f"{details['utterance_idx']}_"
            f"{details['speaker']}_"
            f"d{details['dialog_idx']}.wav"
        )

        rows.append({
            "file_name": wav_file,
            "transcript": details["text"],
            "label": details["act"]
        })

df = pd.DataFrame(rows)

df.to_csv("dataset.csv", index=False)

print(f"Saved {len(df)} rows")