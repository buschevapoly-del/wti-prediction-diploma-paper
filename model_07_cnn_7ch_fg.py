"""
Model 07: Multi-Scale CNN 7 Channels + Fear and Greed
GPU required. Paste 7-channel CNN version from Colab.
Requires: econ_df_1.csv (run data_loading.py first)
"""
import pandas as pd, numpy as np

econ_df_1 = pd.read_csv("econ_df_1.csv", index_col=0, parse_dates=True)
print(f"Data loaded: {econ_df_1.shape}")
# ── Paste your full Colab model code here ─────────────────────────────────────
