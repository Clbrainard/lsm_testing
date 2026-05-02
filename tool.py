import pandas as pd
import shutil
import os

# ── C21-results.csv: add/update runFile column ──────────────────────────────
csv_path = "C21-results.csv"
backup_path = "C21-results.csv.bak"

shutil.copy2(csv_path, backup_path)
df = pd.read_csv(csv_path)
df["runFile"] = (df["N"] >= 200000).astype(int)
df.to_csv(csv_path, index=False)
os.remove(backup_path)
print(f"C21 done. {len(df)} rows updated. Columns: {list(df.columns)}")

# ── C212-results.csv: remove extra column in rows past 6001 ─────────────────
c212_path = "C212-results.csv"
c212_backup = "C212-results.csv.bak"

shutil.copy2(c212_path, c212_backup)

with open(c212_path, "r") as f:
    lines = f.readlines()

header = lines[0]
expected_cols = len(header.strip().split(","))  # 8

fixed_lines = [header]
fixed_count = 0
for line in lines[1:]:
    parts = line.rstrip("\n").split(",")
    if len(parts) == expected_cols + 1:
        # Extra 1 is at index 7 (between dt and fileType); drop it
        parts.pop(7)
        fixed_count += 1
    fixed_lines.append(",".join(parts) + "\n")

with open(c212_path, "w") as f:
    f.writelines(fixed_lines)

os.remove(c212_backup)
print(f"C212 done. {len(fixed_lines)-1} rows processed, {fixed_count} rows fixed.")
