import pandas as pd

# Load CSV
df = pd.read_csv("dataset.csv")

# Keep only rows with label = inform
inform_df = df[df["label"].astype(str).str.lower() == "inform"]

# Save to a new CSV
inform_df.to_csv("declarative.csv", index=False)

print(f"Saved {len(inform_df)} rows to declarative.csv")

