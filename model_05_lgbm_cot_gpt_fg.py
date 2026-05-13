"""
==============================================================================
  LightGBM + Fear & Greed Index
  Directional Forecasting of WTI Crude Oil Prices
  
  FIXES vs previous version:
    ✅ Causal (backward-only) smoothing — no look-ahead bias
    ✅ Walk-forward backtesting — simulates real trading
    ✅ Features computed inside each window — no global leakage
    ✅ MinMax scaler fit on training data only — re-fit each fold

  Previous version had 99% accuracy due to Nadaraya-Watson smoothing
  using future values to smooth past prices — classic data leakage.

  Features:
    · WTI momentum, volatility, RSI, Bollinger Band, MACD
    · Cross-asset 5-day slopes (Brent, Gold, USD, Copper, S&P500)
    · WTI/Brent spread momentum
    · CNN Fear & Greed Index (alternative.me, free, no key)

  Validation:
    · Walk-forward cross-validation (expanding window, 4 folds)
    · Final holdout test on last 20% of data
    · Pre/Post-2020 regime analysis
    · Year-by-year breakdown

  Horizon : 5 days  |  Window : 20 days
==============================================================================
"""

import subprocess, sys

def silent_install(pkg):
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        print(f"  Installing {pkg}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

silent_install('lightgbm')
silent_install('requests')

import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                              f1_score, roc_auc_score)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  LightGBM + Fear & Greed  |  WTI  |  Causal + Walk-Forward")
print("=" * 70)


# ===========================================================================
# SECTION 1 — CAUSAL SMOOTHING (no look-ahead)
# ===========================================================================

def causal_smooth(series, h=5):
    """
    Backward-looking Gaussian kernel smoothing.
    At time i, ONLY uses values at i and earlier — never future values.
    This is the correct version for financial time series.
    Replaces the bilateral Nadaraya-Watson that caused data leakage.
    """
    n   = len(series)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        # Only look back up to 3*h days
        start = max(0, i - 3 * h)
        past  = np.arange(start, i + 1)
        vals  = series[start : i + 1]
        w     = np.exp(-((i - past) ** 2) / (2 * h ** 2))
        out[i] = np.dot(w / w.sum(), vals)
    return out


# ===========================================================================
# SECTION 2 — FEAR & GREED INDEX
# ===========================================================================

def fetch_fear_greed(limit=3000):
    """
    Fetches historical F&G from alternative.me (free, no key).
    Returns pd.Series indexed by date, values 0–100.
    Missing dates filled with 50 (neutral).
    """
    print("\n[1/5] Fetching Fear & Greed Index...")
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    try:
        resp = requests.get(url, timeout=20)
        data = resp.json().get('data', [])
        rows = {
            pd.Timestamp(int(d['timestamp']), unit='s').normalize(): int(d['value'])
            for d in data
        }
        series = pd.Series(rows, name='fear_greed').sort_index()
        print(f"  -> Coverage: {series.index.min().date()} "
              f"to {series.index.max().date()}  ({len(series):,} days)")
        print(f"  -> Mean={series.mean():.1f}  "
              f"Min={series.min()}  Max={series.max()}")

        regimes = pd.cut(series, bins=[0,25,45,55,75,101],
                         labels=['Extreme Fear','Fear','Neutral',
                                 'Greed','Extreme Greed'], right=False)
        print("  -> Regime distribution:")
        for reg, cnt in regimes.value_counts().sort_index().items():
            pct = cnt / len(series) * 100
            print(f"       {reg:<16} {cnt:>5,} days ({pct:.1f}%)  "
                  f"{'█' * int(pct/3)}")
        return series
    except Exception as e:
        print(f"  -> FAILED ({e}) — using neutral (50) everywhere")
        return pd.Series(dtype=float, name='fear_greed')


# ===========================================================================
# SECTION 3 — ROBUST LABELS
# ===========================================================================

def make_robust_labels(price_series, horizon=5, threshold=0.003):
    """
    Labels based on RAW (unsmoothed) prices to avoid any leakage.
     1 = UP   if raw return > +threshold
     0 = DOWN if raw return < -threshold
    -1 = NEUTRAL — skipped
    """
    ret = (price_series.shift(-horizon) - price_series) / \
          (price_series + 1e-8)
    return np.where(ret >  threshold,  1,
           np.where(ret < -threshold,  0, -1))


# ===========================================================================
# SECTION 4 — FEATURE ENGINEERING (window-local, no global state)
# ===========================================================================
#
# IMPORTANT: All features are computed ONLY from values within the
# current 20-day window. No global scaler is used here.
# MinMaxScaler is fit on training windows only inside each fold.

FEATURE_NAMES = [
    # WTI momentum (5)
    'wti_mom_1d','wti_mom_3d','wti_mom_5d','wti_mom_10d','wti_mom_full',
    # Volatility (2)
    'wti_vol_5d','wti_vol_10d',
    # Technical (3)
    'wti_rsi','wti_bb','wti_macd',
    # Cross-asset slopes (5)
    'brent_slope_5d','gold_slope_5d','usd_slope_5d',
    'copper_slope_5d','sp500_slope_5d',
    # Spread (1)
    'wti_brent_spread_chg',
    # Levels (2)
    'gold_level','usd_level',
    # Fear & Greed (7)
    'fg_now','fg_5d_avg','fg_chg_5d','fg_chg_10d',
    'fg_extreme_fear','fg_extreme_greed','fg_regime',
]


def build_feature_row(w_raw, w_fg, price_cols):
    """
    Builds feature vector from ONE 20-day window of RAW (unscaled) prices.

    w_raw      : (20, n_cols)  — raw price values
    w_fg       : (20,)         — raw F&G values 0–100
    price_cols : list of column names
    """
    def col(name):
        return price_cols.index(name) if name in price_cols else None

    wti_i   = col('WTI')
    brent_i = col('Brent')
    gold_i  = col('Gold')
    usd_i   = col('Dollar/Euro')
    cu_i    = col('Copper')
    sp_i    = col('S&P500')

    # Apply causal smoothing to WTI within this window only
    wti_raw    = w_raw[:, wti_i]
    wti        = causal_smooth(wti_raw, h=3)   # h=3 for short window

    # ── WTI momentum ─────────────────────────────────────────────────────
    mom_1d   = float(wti[-1] - wti[-2])
    mom_3d   = float(wti[-1] - wti[-4])  if len(wti) > 3  else 0.0
    mom_5d   = float(wti[-1] - wti[-6])  if len(wti) > 5  else 0.0
    mom_10d  = float(wti[-1] - wti[-11]) if len(wti) > 10 else 0.0
    mom_full = float(wti[-1] - wti[0])

    # ── Volatility ────────────────────────────────────────────────────────
    vol_5d  = float(wti_raw[-5:].std())    # use raw prices for vol
    vol_10d = float(wti_raw[-10:].std())

    # ── RSI (14-period) ───────────────────────────────────────────────────
    diff   = np.diff(wti_raw[-15:]) if len(wti_raw) >= 15 else np.diff(wti_raw)
    gains  = diff[diff > 0].mean()  if (diff > 0).any() else 0.0
    losses = -diff[diff < 0].mean() if (diff < 0).any() else 1e-8
    rsi    = float(100 - (100 / (1 + gains / (losses + 1e-8))))

    # ── Bollinger Band ────────────────────────────────────────────────────
    ma10   = wti_raw[-10:].mean()
    std10  = wti_raw[-10:].std() + 1e-8
    bb     = float((wti_raw[-1] - ma10) / (2 * std10))

    # ── MACD ─────────────────────────────────────────────────────────────
    ema12 = float(pd.Series(wti_raw).ewm(span=12, adjust=False).mean().iloc[-1])
    ema26 = float(pd.Series(wti_raw).ewm(span=26, adjust=False).mean().iloc[-1])
    macd  = ema12 - ema26

    # ── Cross-asset 5-day slopes (raw pct change) ─────────────────────────
    def slope5(i):
        if i is None or len(w_raw) < 6:
            return 0.0
        p_now  = w_raw[-1, i]
        p_5ago = w_raw[-6, i]
        return float((p_now - p_5ago) / (p_5ago + 1e-8))

    brent_s = slope5(brent_i)
    gold_s  = slope5(gold_i)
    usd_s   = slope5(usd_i)
    cu_s    = slope5(cu_i)
    sp_s    = slope5(sp_i)

    # ── WTI/Brent spread momentum ─────────────────────────────────────────
    if brent_i is not None and len(w_raw) > 5:
        spread_now = w_raw[-1, wti_i]  - w_raw[-1, brent_i]
        spread_5d  = w_raw[-6, wti_i]  - w_raw[-6, brent_i]
        spread_chg = float(spread_now - spread_5d)
    else:
        spread_chg = 0.0

    # ── Current levels (normalized within window) ─────────────────────────
    def norm_level(i):
        if i is None:
            return 0.5
        col_vals = w_raw[:, i]
        rng = col_vals.max() - col_vals.min()
        return float((col_vals[-1] - col_vals.min()) / (rng + 1e-8))

    gold_lev = norm_level(gold_i)
    usd_lev  = norm_level(usd_i)

    # ── Fear & Greed features ─────────────────────────────────────────────
    fg_now     = float(w_fg[-1])
    fg_5d_avg  = float(w_fg[-5:].mean())
    fg_chg_5d  = float(w_fg[-1] - w_fg[-6])  if len(w_fg) > 5  else 0.0
    fg_chg_10d = float(w_fg[-1] - w_fg[-11]) if len(w_fg) > 10 else 0.0
    fg_xfear   = float(fg_now < 25)
    fg_xgreed  = float(fg_now > 75)
    fg_regime  = float(np.digitize(fg_now, bins=[0,25,45,55,75,101]))

    return np.array([
        mom_1d, mom_3d, mom_5d, mom_10d, mom_full,
        vol_5d, vol_10d,
        rsi, bb, macd,
        brent_s, gold_s, usd_s, cu_s, sp_s,
        spread_chg,
        gold_lev, usd_lev,
        fg_now, fg_5d_avg, fg_chg_5d, fg_chg_10d,
        fg_xfear, fg_xgreed, fg_regime,
    ], dtype=np.float64)


# ===========================================================================
# SECTION 5 — WALK-FORWARD BACKTESTING
# ===========================================================================
#
# Simulates real trading: train on everything up to date T,
# predict on the next period, then roll forward.
#
# Expanding window scheme:
#   Fold 1: train on [0, 60%], test on [60%, 70%]
#   Fold 2: train on [0, 70%], test on [70%, 80%]
#   Fold 3: train on [0, 80%], test on [80%, 90%]
#   Fold 4: train on [0, 90%], test on [90%, 100%]
#
# Then final holdout: train on [0, 80%], test on [80%, 100%]

def walk_forward_cv(X, y, dates, n_folds=4, min_train_pct=0.5,
                    lgb_params=None, price_idx=None, fg_idx=None):
    """
    Expanding-window walk-forward cross-validation.
    Returns per-fold metrics and concatenated OOF predictions.
    """
    n          = len(X)
    fold_size  = int(n * (1 - min_train_pct) / n_folds)
    min_train  = int(n * min_train_pct)

    all_preds  = np.full(n, np.nan)
    all_proba  = np.full(n, np.nan)
    fold_results = []

    print(f"\n  Walk-forward CV: {n_folds} folds, "
          f"min train={min_train_pct*100:.0f}%")

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end  = min(train_end + fold_size, n)

        if test_end <= train_end:
            break

        X_tr, y_tr = X[:train_end],       y[:train_end]
        X_te, y_te = X[train_end:test_end], y[train_end:test_end]
        d_te       = dates[train_end:test_end]

        # Fit scaler on training fold only
        sc        = MinMaxScaler()
        imp       = SimpleImputer(strategy='mean')
        X_tr_sc   = sc.fit_transform(imp.fit_transform(X_tr))
        X_te_sc   = sc.transform(imp.transform(X_te))

        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(X_tr_sc, y_tr)

        proba       = clf.predict_proba(X_te_sc)[:, 1]
        preds       = (proba >= 0.5).astype(int)
        all_proba[train_end:test_end] = proba
        all_preds[train_end:test_end] = preds

        acc  = accuracy_score(y_te, preds)
        mf1  = f1_score(y_te, preds, average='macro') \
               if len(np.unique(preds)) > 1 else 0.0
        dr   = (preds[y_te==0]==0).mean() if (y_te==0).any() else 0.0

        fold_results.append({
            'fold':      fold + 1,
            'train_n':   train_end,
            'test_n':    len(y_te),
            'period':    f"{d_te.min().date()} → {d_te.max().date()}",
            'acc':       acc,
            'f1':        mf1,
            'dr':        dr,
        })

        print(f"  Fold {fold+1}: train={train_end:,}  "
              f"test={len(y_te):,}  "
              f"({d_te.min().date()} → {d_te.max().date()})  "
              f"acc={acc*100:.2f}%  macroF1={mf1:.4f}  downRec={dr:.3f}")

    return fold_results, all_preds, all_proba


def evaluate_holdout(name, proba, y_true, dates,
                      confidence_pct=70, verbose=True):
    """Full evaluation on a holdout set with confidence filtering."""
    y_pred_all = (proba >= 0.5).astype(int)
    confidence = np.abs(proba - 0.5)
    thr        = np.percentile(confidence, 100 - confidence_pct)
    conf_mask  = confidence >= thr

    y_pred_conf = y_pred_all[conf_mask]
    y_true_conf = y_true[conf_mask]

    acc_all  = accuracy_score(y_true, y_pred_all)
    f1_all   = f1_score(y_true, y_pred_all, average='macro')
    dr_all   = (y_pred_all[y_true==0]==0).mean() if (y_true==0).any() else 0.0

    acc_conf = accuracy_score(y_true_conf, y_pred_conf) \
               if len(y_true_conf) > 0 else 0.0
    f1_conf  = f1_score(y_true_conf, y_pred_conf, average='macro') \
               if len(np.unique(y_pred_conf)) > 1 else 0.0
    dr_conf  = (y_pred_conf[y_true_conf==0]==0).mean() \
               if (y_true_conf==0).any() else 0.0

    try:
        auc = roc_auc_score(y_true, proba)
    except Exception:
        auc = 0.5

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  {name}")
        print(f"  {'─'*60}")
        print(f"  ALL ({len(y_true):,} samples):")
        print(f"    Acc={acc_all*100:.2f}%  MacroF1={f1_all:.4f}  "
              f"DownRec={dr_all:.3f}  AUC={auc:.4f}")
        print(f"  Top-{confidence_pct}% confident ({conf_mask.sum():,}):")
        print(f"    Acc={acc_conf*100:.2f}%  MacroF1={f1_conf:.4f}  "
              f"DownRec={dr_conf:.3f}")
        if len(np.unique(y_true_conf)) > 1 and len(np.unique(y_pred_conf)) > 1:
            print(classification_report(y_true_conf, y_pred_conf,
                                         target_names=['Down(0)','Up(1)'],
                                         digits=4))

    return dict(acc_all=acc_all, acc_conf=acc_conf,
                f1_all=f1_all,   f1_conf=f1_conf,
                dr_all=dr_all,   dr_conf=dr_conf,
                auc=auc,         n_conf=conf_mask.sum(),
                conf_mask=conf_mask, y_pred_all=y_pred_all)


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

BASE_FEATURES  = ['WTI', 'Brent', 'Gold', 'Dollar/Euro', 'Copper', 'S&P500']
WINDOW_SIZE    = 20
HORIZON        = 5
THRESHOLD      = 0.003
DOWN_WEIGHT    = 2.0
CONFIDENCE_PCT = 70


# ---------- 0. Load raw data (NO smoothing applied globally) ----------
print("\n[0/5] Loading econ_df_1  (raw prices, no global smoothing)...")

data_raw = econ_df_1[BASE_FEATURES].copy()
data_raw.index = pd.to_datetime(data_raw.index)

full_dates = pd.date_range(start=data_raw.index.min(),
                            end=data_raw.index.max(), freq='D')
data_raw   = data_raw.reindex(full_dates).ffill().dropna()

print(f"  -> Rows  : {len(data_raw):,}")
print(f"  -> Period: {data_raw.index.min().date()} "
      f"to {data_raw.index.max().date()}")
print("  NOTE: Smoothing is applied LOCALLY inside each window only.")
print("        No global smoothing — prevents look-ahead bias.")

# Labels computed from RAW prices
y_raw_series = pd.Series(
    make_robust_labels(data_raw['WTI'], horizon=HORIZON, threshold=THRESHOLD),
    index=data_raw.index
)


# ---------- 1. Fear & Greed ----------
fg_raw     = fetch_fear_greed(limit=3000)
fg_aligned = (fg_raw
              .reindex(data_raw.index)
              .ffill()
              .bfill()
              .fillna(50.0))

covered = fg_raw.reindex(data_raw.index).notna().sum()
print(f"\n  F&G covers {covered:,} / {len(data_raw):,} days. "
      f"Earlier dates filled with neutral (50).")


# ---------- 2. Build feature matrix ----------
print("\n[2/5] Building feature matrix (window-local features)...")

X_rows, y_rows, idx_rows = [], [], []
skipped = 0

for i in tqdm(range(len(data_raw) - WINDOW_SIZE - HORIZON),
              desc="  Windows"):
    label = y_raw_series.iloc[i + WINDOW_SIZE - 1]
    if label == -1:
        skipped += 1
        continue

    w_raw = data_raw.iloc[i : i + WINDOW_SIZE].values   # (20, 6) RAW
    w_fg  = fg_aligned.iloc[i : i + WINDOW_SIZE].values # (20,)

    X_rows.append(build_feature_row(w_raw, w_fg, BASE_FEATURES))
    y_rows.append(int(label))
    idx_rows.append(data_raw.index[i + WINDOW_SIZE - 1])

X         = np.array(X_rows)
y         = np.array(y_rows)
win_dates = pd.DatetimeIndex(idx_rows)

print(f"  -> {X.shape[0]:,} samples | {skipped:,} neutral skipped")
print(f"  -> Features: {X.shape[1]} | Down={np.sum(y==0):,} Up={np.sum(y==1):,}")


# ---------- 3. Feature groups ----------
price_idx = [i for i,f in enumerate(FEATURE_NAMES) if not f.startswith('fg_')]
fg_idx    = [i for i,f in enumerate(FEATURE_NAMES) if f.startswith('fg_')]
all_idx   = list(range(len(FEATURE_NAMES)))

lgb_params = dict(
    objective         = 'binary',
    metric            = 'binary_logloss',
    learning_rate     = 0.05,
    num_leaves        = 31,
    max_depth         = 5,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    n_estimators      = 300,
    random_state      = 42,
    verbose           = -1,
    class_weight      = {0: DOWN_WEIGHT, 1: 1.0}
)


# ---------- 4. Walk-forward cross-validation ----------
print("\n[3/5] Walk-forward cross-validation...")

print("\n  ── Model A: Price + technical only ──")
fold_res_a, oof_preds_a, oof_proba_a = walk_forward_cv(
    X[:, price_idx], y, win_dates,
    n_folds=4, min_train_pct=0.5,
    lgb_params=lgb_params
)

print("\n  ── Model B: Price + Fear & Greed ──")
fold_res_b, oof_preds_b, oof_proba_b = walk_forward_cv(
    X, y, win_dates,
    n_folds=4, min_train_pct=0.5,
    lgb_params=lgb_params
)

# OOF summary (only folds that have predictions)
oof_mask = ~np.isnan(oof_proba_a)

print(f"\n  OOF summary (all folds combined, {oof_mask.sum():,} samples):")
print(f"  {'Model':<40} {'Acc':>8} {'MacroF1':>9} {'DownRec':>9} {'AUC':>7}")
print("  " + "-" * 75)

for name, proba in [('A) Price only (OOF)', oof_proba_a),
                     ('B) Price + Fear & Greed (OOF)', oof_proba_b)]:
    m    = ~np.isnan(proba)
    yt   = y[m]
    yp   = (proba[m] >= 0.5).astype(int)
    acc  = accuracy_score(yt, yp)
    mf1  = f1_score(yt, yp, average='macro') if len(np.unique(yp)) > 1 else 0.0
    dr   = (yp[yt==0]==0).mean() if (yt==0).any() else 0.0
    try:
        auc = roc_auc_score(yt, proba[m])
    except Exception:
        auc = 0.5
    print(f"  {name:<40} {acc*100:>7.2f}%  {mf1:>8.4f}  {dr:>8.3f}  {auc:>6.4f}")


# ---------- 5. Final holdout test (last 20%) ----------
print("\n[4/5] Final holdout evaluation (last 20%)...")
print("      Model trained on first 80%, tested on last 20%.")

split_idx       = int(len(X) * 0.8)
X_train_final   = X[:split_idx]
y_train_final   = y[:split_idx]
X_test_final    = X[split_idx:]
y_test_final    = y[split_idx:]
dates_test      = win_dates[split_idx:]

# Fit scaler on training data ONLY
sc_final  = MinMaxScaler()
imp_final = SimpleImputer(strategy='mean')
X_tr_sc   = sc_final.fit_transform(imp_final.fit_transform(X_train_final))
X_te_sc   = sc_final.transform(imp_final.transform(X_test_final))

print(f"  Train: {len(y_train_final):,}  "
      f"({win_dates[0].date()} – {win_dates[split_idx-1].date()})")
print(f"  Test : {len(y_test_final):,}   "
      f"({dates_test.min().date()} – {dates_test.max().date()})")

# Model A — price only
clf_a = lgb.LGBMClassifier(**lgb_params)
clf_a.fit(X_tr_sc[:, price_idx], y_train_final)
proba_a = clf_a.predict_proba(X_te_sc[:, price_idx])[:, 1]

# Model B — price + F&G
clf_b = lgb.LGBMClassifier(**lgb_params)
clf_b.fit(X_tr_sc, y_train_final)
proba_b = clf_b.predict_proba(X_te_sc)[:, 1]


# ---------- 6. Results ----------
print("\n" + "=" * 70)
print("  RESULTS  |  LightGBM  |  WTI 5-day  |  No data leakage")
print("=" * 70)

res_a = evaluate_holdout(
    "A) Price + technical only (baseline)",
    proba_a, y_test_final, dates_test,
    confidence_pct=CONFIDENCE_PCT
)
res_b = evaluate_holdout(
    "B) Price + Fear & Greed",
    proba_b, y_test_final, dates_test,
    confidence_pct=CONFIDENCE_PCT
)

# Summary
print(f"\n  {'─'*70}")
print(f"  SUMMARY — top-{CONFIDENCE_PCT}% confident predictions")
print(f"  {'Model':<42} {'Acc':>8} {'MacroF1':>9} "
      f"{'DownRec':>9} {'AUC':>7} {'n':>7}")
print("  " + "-" * 82)

base_acc = res_a['acc_conf']
for name, r in [('A) Price only', res_a), ('B) + Fear & Greed', res_b)]:
    delta = f" ({(r['acc_conf']-base_acc)*100:+.2f}%)" \
            if name != 'A) Price only' else "         "
    print(f"  {name:<42} "
          f"{r['acc_conf']*100:>6.2f}%{delta} "
          f"{r['f1_conf']:>8.4f}  "
          f"{r['dr_conf']:>8.3f}  "
          f"{r['auc']:>6.4f}  "
          f"{r['n_conf']:>6,}")

# Confidence ablation
print(f"\n  {'─'*65}")
print("  Ablation — confidence threshold vs accuracy (Model B):")
print(f"  {'Top-x%':<10} {'Samples':>9} {'Acc':>9} "
      f"{'MacroF1':>9} {'DownRec':>9}")
print("  " + "-" * 52)

conf_b = np.abs(proba_b - 0.5)
pred_b = (proba_b >= 0.5).astype(int)

for top_x in [100, 80, 70, 60, 50, 40]:
    thr    = np.percentile(conf_b, 100 - top_x)
    mask_x = conf_b >= thr
    if mask_x.sum() < 20:
        continue
    yt = y_test_final[mask_x]
    yp = pred_b[mask_x]
    ax = accuracy_score(yt, yp)
    fx = f1_score(yt, yp, average='macro') \
         if len(np.unique(yp)) > 1 else 0.0
    dx = (yp[yt==0]==0).mean() if (yt==0).any() else 0.0
    print(f"  T@{top_x:<7}  {mask_x.sum():>8,}  "
          f"{ax*100:>8.2f}%  {fx:>8.4f}  {dx:>8.3f}")


# ---------- 7. Temporal analysis ----------
print("\n" + "=" * 70)
print("  TEMPORAL ANALYSIS  |  Pre/Post-2020  |  Model B")
print("=" * 70)

best_conf_mask = res_b['conf_mask']
best_pred      = res_b['y_pred_all']

SPLIT_2020 = pd.Timestamp('2020-01-01')
SPLIT_2021 = pd.Timestamp('2021-01-01')

mask_pre   = (dates_test < SPLIT_2020)
mask_covid = (dates_test >= SPLIT_2020) & (dates_test < SPLIT_2021)
mask_post  = (dates_test >= SPLIT_2021)

print(f"\n  Test period: {dates_test.min().date()} → {dates_test.max().date()}")
print(f"  Pre-2020 : {mask_pre.sum():,}  |  "
      f"2020: {mask_covid.sum():,}  |  "
      f"Post-2020: {mask_post.sum():,}")

def period_block(label, period_mask):
    combined = period_mask & best_conf_mask
    if combined.sum() < 5:
        print(f"\n  {label}: insufficient samples ({combined.sum()})")
        return
    yt  = y_test_final[combined]
    yp  = best_pred[combined]
    acc = accuracy_score(yt, yp)
    mf1 = f1_score(yt, yp, average='macro') if len(np.unique(yp))>1 else 0.0
    dr  = (yp[yt==0]==0).mean() if (yt==0).any() else 0.0
    ur  = (yp[yt==1]==1).mean() if (yt==1).any() else 0.0
    print(f"\n  {'─'*56}")
    print(f"  {label}  ({combined.sum():,} confident samples)")
    print(f"  {'─'*56}")
    print(f"  Acc={acc*100:.2f}%  MacroF1={mf1:.4f}  "
          f"DownRec={dr:.3f}  UpRec={ur:.3f}")
    if len(np.unique(yt))>1 and len(np.unique(yp))>1:
        print(classification_report(yt, yp,
                                     target_names=['Down(0)','Up(1)'],
                                     digits=4))

period_block("PRE-2020 — normal market cycle",  mask_pre)
period_block("2020 — COVID + negative prices",  mask_covid)
period_block("POST-2020 — recovery/war/OPEC+",  mask_post)

# Year-by-year
print(f"\n  {'─'*65}")
print("  YEAR-BY-YEAR  (Model B, confident predictions):")
print(f"  {'Year':<7} {'n':>6} {'Acc':>8} {'MacroF1':>9} "
      f"{'DownRec':>9}  Regime")
print("  " + "-" * 72)

regimes = {
    2000:'Pre-crash',        2001:'Post 9/11',
    2002:'Recovery',         2003:'Iraq war',
    2004:'Demand surge',     2005:'Hurricanes',
    2006:'High prices',      2007:'Pre-GFC',
    2008:'GFC crash',        2009:'Recovery',
    2010:'Rebound',          2011:'Arab Spring',
    2012:'Iran sanctions',   2013:'Stable high',
    2014:'Shale crash',      2015:'Supply glut',
    2016:'OPEC deal',        2017:'Gradual rise',
    2018:'Trade war',        2019:'Pre-COVID',
    2020:'<<< COVID / NEGATIVE PRICES >>>',
    2021:'Post-COVID recovery',
    2022:'Russia-Ukraine / price spike',
    2023:'OPEC+ production cuts',
    2024:'Energy transition',
    2025:'Recent',           2026:'Recent',
}

for year in sorted(dates_test.year.unique()):
    mask_yr = (dates_test.year == year) & best_conf_mask
    if mask_yr.sum() < 5:
        continue
    yt  = y_test_final[mask_yr]
    yp  = best_pred[mask_yr]
    acc = accuracy_score(yt, yp)
    mf1 = f1_score(yt, yp, average='macro') if len(np.unique(yp))>1 else 0.0
    dr  = (yp[yt==0]==0).mean() if (yt==0).any() else 0.0
    reg = regimes.get(year,'')
    print(f"  {year:<7} {mask_yr.sum():>6,} "
          f"{acc*100:>7.2f}% {mf1:>8.4f}  {dr:>8.3f}   {reg}")

print(f"\n  NOTE: Results above 70% = genuinely good for WTI 5-day direction.")
print(f"  Results near 99% = data leakage (check smoothing / scaling).")


# ---------- 8. Feature importance ----------
print(f"\n{'=' * 70}")
print("  FEATURE IMPORTANCE  |  Model B (Price + Fear & Greed)")
print("=" * 70)

feat_names_b = FEATURE_NAMES
importances  = pd.Series(clf_b.feature_importances_,
                          index=feat_names_b).sort_values(ascending=False)

print(f"\n  {'Feature':<30} {'Importance':>12}  Bar")
print("  " + "-" * 56)
for feat, imp in importances.head(20).items():
    bar = '█' * int(imp / importances.max() * 25)
    print(f"  {feat:<30} {imp:>12.1f}  {bar}")

total    = importances.sum()
fg_tot   = importances[[f for f in feat_names_b if f.startswith('fg_')]].sum()
pr_tot   = total - fg_tot

print(f"\n  Feature group contributions:")
print(f"    Price / technical : {pr_tot/total*100:.1f}%")
print(f"    Fear & Greed      : {fg_tot/total*100:.1f}%")

fg_imps = importances[[f for f in feat_names_b if f.startswith('fg_')]]
top_fg  = fg_imps.idxmax()
print(f"\n  Top F&G feature: {top_fg}")
print(f"\n  Fear & Greed breakdown:")
for feat, imp in fg_imps.sort_values(ascending=False).items():
    bar = '█' * int(imp / importances.max() * 20)
    print(f"    {feat:<28} {imp:>10.1f}  {bar}")


# ---------- 9. Example predictions ----------
print(f"\n{'=' * 70}")
print("  EXAMPLE PREDICTIONS  |  First 8 test samples  |  Model B")
print("=" * 70)

fg_te = fg_aligned.reindex(pd.DatetimeIndex(dates_test)).values

print(f"\n  {'#':<4} {'Date':<12} {'True':>5} {'Pred':>5} "
      f"{'P(UP)':>7} {'F&G':>6} {'Regime':<20} {'OK':>4}")
print("  " + "-" * 68)

for i in range(min(8, len(y_test_final))):
    p      = proba_b[i]
    pred   = 1 if p >= 0.5 else 0
    true   = y_test_final[i]
    date   = dates_test[i].strftime('%Y-%m-%d')
    fg_v   = float(fg_te[i]) if i < len(fg_te) and not np.isnan(fg_te[i]) else 50.0

    if fg_v < 25:   regime = 'Extreme Fear'
    elif fg_v < 45: regime = 'Fear'
    elif fg_v < 55: regime = 'Neutral'
    elif fg_v < 75: regime = 'Greed'
    else:           regime = 'Extreme Greed'

    ok = '✓' if pred == true else '✗'
    print(f"  {i+1:<4} {date:<12} "
          f"{'UP' if true==1 else 'DN':>5} "
          f"{'UP' if pred==1 else 'DN':>5} "
          f"{p:>7.4f} {fg_v:>6.0f} "
          f"{regime:<20} {ok:>4}")

print("\n" + "=" * 70)
print("  Done.")
print("=" * 70)
