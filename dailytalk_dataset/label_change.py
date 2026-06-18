import pandas as pd

with open ("dataset.csv", "r", encoding="utf-8") as f:
    data = pd.read_csv(f)

df = pd.DataFrame(data)

label_map = {
    "inform": "statement"
}

df["label"] = df["label"].replace(label_map)

df.to_csv("updated_dataset.csv", index=False)

print("Labels updated successfully!")