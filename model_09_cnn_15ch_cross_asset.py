"""
Model 09: Multi-Scale CNN 15 Channels + Cross-Asset Features
GPU required.
Requires: econ_df_1.csv (run data_loading.py first)
"""
import pandas as pd, numpy as np

econ_df_1 = pd.read_csv("econ_df_1.csv", index_col=0, parse_dates=True)
print(f"Data loaded: {econ_df_1.shape}")
# ── Paste your full Colab model code here ─────────────────────────────────────
