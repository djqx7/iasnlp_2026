import pandas as pd

df = pd.read_csv("file1.csv")

result_df = (
    df.groupby("type", group_keys=False)
      .head(100)
)

result_df.to_csv("sample_100_per_type.csv", index=False)

print(result_df["type"].value_counts())
