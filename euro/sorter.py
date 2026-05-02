import pandas as pd

df = pd.read_csv("euro/EuroConv_sorted.csv")

counts = df["N"].value_counts().sort_index()
print(counts.to_string())
