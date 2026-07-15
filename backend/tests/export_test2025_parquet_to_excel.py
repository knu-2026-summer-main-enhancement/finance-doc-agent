from pathlib import Path

import pandas as pd


dataframe_dir = Path(__file__).resolve().parents[1] / "dataframes"
source = dataframe_dir / "df_test2025_img_table0.parquet"
destination = dataframe_dir / "df_test2025_img_table0.xlsx"

df = pd.read_parquet(source)
df.to_excel(destination, index=False, engine="openpyxl")

exported = pd.read_excel(destination, engine="openpyxl")
if exported.shape != df.shape:
    raise RuntimeError(f"shape mismatch: parquet={df.shape}, excel={exported.shape}")

print(f"exported={destination}")
print(f"shape={exported.shape}")
