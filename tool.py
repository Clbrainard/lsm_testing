import pandas as pd
import shutil
import os

csv_path = "C21-results.csv"
backup_path = "C21-results.csv.bak"

shutil.copy2(csv_path, backup_path)

df = pd.read_csv(csv_path)

df["runFile"] = (df["N"] >= 200000).astype(int)

df.to_csv(csv_path, index=False)

os.remove(backup_path)

print(f"Done. {len(df)} rows updated. Columns: {list(df.columns)}")
