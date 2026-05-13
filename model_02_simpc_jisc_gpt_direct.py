"""
Model 02: SIMPC + JISC-Net + LGB + GPT Direct Sentiment
OpenAI API key required. Paste GPT direct version from Colab.
Requires: econ_df_1.csv (run data_loading.py first)
"""
import pandas as pd, numpy as np

econ_df_1 = pd.read_csv("econ_df_1.csv", index_col=0, parse_dates=True)
print(f"Data loaded: {econ_df_1.shape}")
# ── Paste your full Colab model code here ─────────────────────────────────────
