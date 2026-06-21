import pandas as pd

# Read CSV
df = pd.read_csv("dataset.csv")

# Select rows where transcript ends with ?
sentences_with_interrogative = df[
    df["transcript"].astype(str).str.contains(r'\?\s*$', regex=True, na=False)
]

# Count
count = len(sentences_with_interrogative)

print(sentences_with_interrogative)
print("Number of sentences ending with '?':", count)

# Save to a new CSV file
sentences_with_interrogative.to_csv(
    "interrogative.csv",
    index=False
)

print("Saved to sentences_ending_with_interrogative.csv")

