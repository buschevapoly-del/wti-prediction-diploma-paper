# WTI Crude Oil Directional Forecasting

Master's Thesis — Directional Forecasting of WTI Crude Oil Prices  
using Pattern Recognition, LLM Sentiment and CNN Fear & Greed Index

---

## Results

| File | Model | Training | Trades | Post-2020 Acc | Sharpe |
|------|-------|----------|--------|---------------|--------|
| — | Random walk baseline | — | — | 50.00% | — |
| model_01 | SIMPC + JISC-Net + LGB: Price only | 2000–2026 | 288 | 55.56% | 0.294 |
| model_02 | SIMPC + JISC-Net + LGB: + GPT direct | 2000–2026 | 259 | 56.76% | 0.162 |
| model_03 | SIMPC + JISC-Net + LGB: + CoT GPT | 2000–2026 | 278 | 55.76% | 0.182 |
| model_04 | SIMPC + JISC-Net + LGB: + Fear & Greed | 2014–2026 | 344 | 50.58% | 0.423 |
| **model_05** | **LightGBM + CoT GPT + Fear & Greed** | **2020–2026** | **215** | **57.67%** | **0.532** |
| model_06 | CNN 4ch (WTI + RSI + MACD + Vol) | 2020–2026 | 115 | 53.04% | 0.423 |
| model_07 | CNN 7ch + Fear & Greed | 2020–2026 | 102 | 51.96% | 0.026 |
| model_08 | CNN 15ch + CoT GPT | 2020–2026 | 180 | 47.78% | −0.824 |
| model_09 | CNN 15ch + cross-asset | 2000–2026 | 107 | 46.73% | −0.664 |
| model_10 | CNN Fear & Greed fusion (74 folds) | 2013–2026 | 603 | 47.93% | 0.232 |
| model_11 | Pure Fear & Greed contrarian signal | 2020–2026 | 211 | 54.50% | — |

**Model 05** is the best: highest post-2020 accuracy (57.67%) and Sharpe ratio (0.532).

---

## Files

| File | Description | Needs |
|------|-------------|-------|
| `data_loading.py` | Downloads FRED + Yahoo Finance data | — |
| `model_01_simpc_jisc_price_only.py` | SIMPC + JISC-Net + LightGBM, price features | — |
| `model_02_simpc_jisc_gpt_direct.py` | + direct GPT sentiment | OpenAI key |
| `model_03_simpc_jisc_cot_gpt.py` | + Chain-of-Thought GPT sentiment | OpenAI key |
| `model_04_simpc_jisc_fg.py` | + CNN Fear & Greed Index | — |
| `model_05_lgbm_cot_gpt_fg.py` | LightGBM + CoT GPT + F&G — **BEST** | OpenAI key |
| `model_06_cnn_4ch.py` | Multi-scale CNN, 4 channels | GPU |
| `model_07_cnn_7ch_fg.py` | CNN 7ch + Fear & Greed | GPU |
| `model_08_cnn_15ch_cot_gpt.py` | CNN 15ch + CoT GPT | GPU + OpenAI key |
| `model_09_cnn_15ch_cross_asset.py` | CNN 15ch + cross-asset | GPU |
| `model_10_cnn_fg_fusion.py` | CNN + F&G fusion, 74 folds | GPU |
| `model_11_fg_contrarian.py` | Pure F&G contrarian rule, no ML | — |

---

## Setup

```bash
pip install -r requirements.txt
python data_loading.py                    # saves econ_df_1.csv
python model_05_lgbm_cot_gpt_fg.py       # run best model
```

CNN models (06–10) require GPU — run in Google Colab with T4.  
Models 02, 03, 05, 08 require an OpenAI API key.

---

## Key Methodological Notes

- **Causal smoothing** — backward-looking kernel only, prevents look-ahead bias
- **Two-track features** — 4 decorrelated features for SIMPC, full set for LightGBM
- **Walk-forward backtest** — expanding window, no static split
- **Horizon** — 5 trading days ahead
- **Label threshold** — ±0.3% neutral zone removes noise labels

---

## References

Kim et al. (2025) — *From Patterns to Predictions: A Shapelet-Based Framework  
for Directional Forecasting in Noisy Financial Markets*. CIKM 2025.  
https://doi.org/10.1145/3746252.3761250

Jiang, Kelly & Xiu (2023) — *Re-Imagining Price Trends*. Journal of Finance.  
https://github.com/lich99/Stock_CNN
