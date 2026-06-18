import pandas as pd

# Read the CSV file
df = pd.read_csv("file1.csv")

# Assuming:
# Column 1 = sentence
# Column 2 = label

# Count occurrences of each label
label_counts = df.iloc[:, 1].value_counts()

# Print the counts
print(label_counts)

# Optional: save counts to a CSV file
label_counts.reset_index().rename(
    columns={"index": "label", df.columns[1]: "count"}
).to_csv("label_counts.csv", index=False)