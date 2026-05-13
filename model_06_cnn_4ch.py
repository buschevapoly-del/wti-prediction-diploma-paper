import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

series_list = [
    'DCOILWTICO',      # WTI
    'DEXUSEU',         # Dollar/Euro
    'DJIA',            # Dow Jones
    'NASDAQCOM',       # NASDAQ
    'DCOILBRENTEU',    # Brent
    'CPIENGSL',        # US CPI Energy
    'CPIAUCSL',        # US CPI Total
    'PPIACO',          # US PPI
    'FEDFUNDS',        # Fed Funds Rate
    'PIEAEN01EZM661N'  # EU PPI
]

start = datetime(2000, 1, 1)
end = datetime.now()

def get_fred_data(series_list, start, end):
    fred_df = pd.DataFrame()
    for i, series in enumerate(series_list):
        print(f'📥 Downloading: {series}')
        success = False
        for attempt in range(3):          # retry up to 3 times
            try:
                if i == 0:
                    fred_df = web.get_data_fred(series, start, end)
                else:
                    _df = web.get_data_fred(series, start, end)
                    fred_df = fred_df.join(_df, how='outer')
                success = True
                break
            except Exception as e:
                print(f"   Attempt {attempt+1} failed: {e}")
                time.sleep(2)             # wait 2 seconds before retry
        if not success:
            print(f"❌ Failed to download {series}")
    return fred_df

# Run
econ_df = get_fred_data(series_list, start, end)

# Rename columns
econ_df.columns = ['WTI', 'Dollar/Euro', 'DJ', 'NASDAQ', 'Brent',
                   'US_CPI_energy', 'US_CPI_total', 'US_PPI_total',
                   'Fed_Funds_Effective', 'EU_PPI']

print("✅ Done! Last 5 rows:")
print(econ_df.tail())

econ_df.reset_index(inplace=True)
econ_df['Date'] = pd.to_datetime(econ_df['DATE'], errors='coerce')
econ_df.loc[:, 'Date'] = econ_df.loc[:, 'Date'].dt.strftime("%Y-%m-%d")
econ_df.index = econ_df['Date']
del econ_df['Date']
del econ_df['DATE']
econ_df.tail(10)

import yfinance as yf
import pandas as pd
from datetime import datetime

# Date range
startDate = datetime(2000, 1, 1)
endDate = datetime.now()

# Symbols you want to download
symbols = ['^SPX', 'DJI за', 'HG=F', 'GC=F', 'NG=F', 'BTC-USD']

# Download all data at once (much faster)
data = yf.download(symbols, start=startDate, end=endDate, progress=False)['Close']

# Rename columns to clear names
data.columns = ['S&P500', 'Dow_Jones', 'Copper', 'Gold', 'Natural_Gas', 'Bitcoin']

# Make sure index is datetime
data.index = pd.to_datetime(data.index)

# Work on a copy of your existing dataframe
econ_df_1 = econ_df.copy()

# If your econ_df has a 'Date' column, convert it to index
if 'Date' in econ_df_1.columns:
    econ_df_1['Date'] = pd.to_datetime(econ_df_1['Date'])
    econ_df_1.set_index('Date', inplace=True)

# Join the new data
econ_df_1 = econ_df_1.join(data, how='left')

print("✅ Successfully added yfinance data!")
print(econ_df_1.tail(10))

econ_df_1 = econ_df_1.rename(columns={'^SPX': 'S&P500', 'HG=F': 'Copper', 'GC=F': 'Gold', 'NG=F': 'Gas', "BTC-USD": 'Bitcoin'})
econ_df_1
econ_df_1.index = pd.to_datetime(econ_df.index)

for col in ["US_CPI_energy", "US_CPI_total", "US_PPI_total",
                   "Fed_Funds_Effective", 'EU_PPI']:

    econ_df_1[col].fillna(method='ffill', inplace=True)
econ_df_1.tail(100)

# 1. Make sure the index is a proper daily datetime index
econ_df_1.index = pd.to_datetime(econ_df_1.index)

# 2. Create a complete daily date range (from 2000 to today)
full_date_range = pd.date_range(start='2000-01-01', end=econ_df_1.index.max(), freq='D')

# 3. Reindex the dataframe to have every single day
econ_df_clean = econ_df_1.reindex(full_date_range)

# 4. Fill missing values:
#    - Forward fill (most important for prices and indices)
#    - Backward fill only for the very beginning (if needed)
econ_df_clean = econ_df_clean.ffill()      # carry forward last known value
econ_df_clean = econ_df_clean.bfill()      # fill any remaining NaNs at the start

# 5. Check the result
print("NaN count per column after cleaning:")
print(econ_df_clean.isna().sum())

print("\nLast 10 rows:")
print(econ_df_clean.tail(10))

# Check correlation with WTI (higher = more useful)
corr = econ_df_1.corr()['WTI'].sort_values(ascending=False)
print("Feature correlation with WTI:")
print(corr)

# Optional: Drop low-correlation features
low_corr_features = corr[abs(corr) < 0.15].index.tolist()
print(f"\nLow-correlation features (consider removing): {low_corr_features}")

econ_df_1 = econ_df_1.drop(columns=['Dow_Jones'], errors='ignore')

pip install tslearn

import pandas as pd
import numpy as np

print("🔄 Handling NaNs in econ_df_1...\n")

# 1. Make sure index is datetime
econ_df_1.index = pd.to_datetime(econ_df_1.index)

# 2. Create complete daily date range
full_dates = pd.date_range(start=econ_df_1.index.min(),
                           end=econ_df_1.index.max(),
                           freq='D')

# 3. Reindex and fill NaNs
econ_df_clean = econ_df_1.reindex(full_dates)

# 4. Fill NaNs (best practice for financial data)
econ_df_clean = econ_df_clean.ffill()      # forward fill (most important)
econ_df_clean = econ_df_clean.bfill()      # backward fill for very first rows only

# 5. Summary
nan_before = econ_df_1.isna().sum().sum()
nan_after  = econ_df_clean.isna().sum().sum()

print(f"✅ NaN handling completed!")
print(f"   Rows before: {len(econ_df_1)}")
print(f"   Rows after (complete daily): {len(econ_df_clean)}")
print(f"   NaNs before: {nan_before}")
print(f"   NaNs after : {nan_after}")

if nan_after == 0:
    print("   → All NaNs successfully removed!")
else:
    print("   → Some NaNs remain (check manually)")

# Optional: Show which columns still have NaNs
if nan_after > 0:
    print("\nColumns with remaining NaNs:")
    print(econ_df_clean.isna().sum()[econ_df_clean.isna().sum() > 0])

# ========================
# Replace original with cleaned version
# ========================
econ_df_1 = econ_df_clean.copy()

print("\n✅ econ_df_1 is now cleaned and ready for modeling!")

"""
==============================================================================
  Multi-Scale CNN + LightGBM  |  WTI Crude Oil  |  Walk-Forward Backtest
  Based on: Jiang, Kelly & Xiu (2023) — "(Re-)Imag(in)ing Price Trends"
  Code:     github.com/lich99/Stock_CNN

  This implementation follows the original paper as closely as possible.
  Differences from the original that are unavoidable for WTI adaptation:
    1. Walk-forward backtest instead of single train/test split
       (necessary because we have one asset, not thousands of stocks)
    2. Three CNN branches concatenated into LightGBM
       (professor's multi-scale extension; original trains three separate models)
    3. Label threshold ±0.3% to remove near-zero moves
       (original uses simple Ret > 0 on stocks with many observations)

  Everything else matches the original exactly:
    · CNN architecture: Conv2d(1→64→128→256, kernel=(5,3), dilation=(2,1),
                        stride=(3,1), padding=(12,1)) + BN + LeakyReLU(0.01)
                        + MaxPool2d(2,1)
    · Loss:       CrossEntropyLoss() — no class weighting
    · Optimizer:  Adam(lr=1e-5)
    · Early stop: patience = 5 epochs, max = 100 epochs
    · Batch size: 128 train / 256 validation
    · Init:       Xavier uniform (Conv2d and Linear)
    · Threshold:  0.58 confidence (from test.ipynb)
    · Images:     1 pixel per price level, uint8 0/255, H×W greyscale
    · Image sizes: 5d→(32,15)  20d→(64,60)  60d→(96,180)
==============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'lightgbm', '-q'])
    import lightgbm as lgb

print("=" * 70)
print("  Multi-Scale CNN + LightGBM  |  WTI  |  Walk-Forward Backtest")
print("  Architecture: Jiang, Kelly & Xiu (2023) — Re-Imagining Price Trends")
print("=" * 70)


# ===========================================================================
# CONFIGURATION — matches original paper where possible
# ===========================================================================

# Image sizes — exact from train.ipynb: IMAGE_HEIGHT / IMAGE_WIDTH dicts
IMAGE_SIZE  = {5: (32, 15), 20: (64, 60), 60: (96, 180)}
WINDOWS     = [5, 20, 60]

# Training — exact from train.ipynb
LOSS_FN     = nn.CrossEntropyLoss()    # no class weighting (original)
LR          = 1e-5                      # Adam lr=1e-5 (original)
MAX_EPOCHS  = 100                       # epochs=100 (original)
EARLY_STOP  = 5                         # early_stopping_epoch=5 (original)
BATCH_TRAIN = 128                       # batch_size=128 (original)
BATCH_VAL   = 256                       # eval batch_size=256 (original)
TRAIN_RATIO = 0.70                      # train_val_ratio=0.7 (original)

# Prediction — exact from test.ipynb
CONF_THRESHOLD = 0.58                   # threshold=0.58 (original)

# WTI-specific adaptations
FEATURES    = ['WTI', 'Brent', 'Gold', 'Dollar/Euro', 'Copper', 'S&P500']
HORIZON     = 5
THRESHOLD   = 0.003                     # ±0.3% label threshold

# Walk-forward (necessary adaptation — one asset not thousands)
INIT_TRAIN  = 0.50
STEP_SIZE   = 60
MIN_TRAIN   = 300
TRADE_FEE   = 0.001

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

print(f"  Device : {DEVICE}")
print(f"  Images : {IMAGE_SIZE}")
print(f"  lr={LR}  epochs={MAX_EPOCHS}  early_stop={EARLY_STOP}  threshold={CONF_THRESHOLD}")


# ===========================================================================
# PRICE → 2D IMAGE
# Matches original: 1 pixel per price level, uint8 (0 or 255).
# High price = top row (chart convention).
# ===========================================================================

def price_to_image(price_window, height, width):
    """
    Converts 1D price series (length=width) to 2D uint8 image (H×W).
    Exactly one pixel is set per time step, at the discretised price level.
    Pixel value = 255 (white), background = 0 (black).
    Same encoding as the original .dat files in Stock_CNN.
    """
    p_min = price_window.min()
    p_max = price_window.max()
    p_norm = (price_window - p_min) / (p_max - p_min + 1e-8)
    img = np.zeros((height, width), dtype=np.uint8)
    for t, pn in enumerate(p_norm):
        row = int(np.clip(pn * (height - 1), 0, height - 1))
        img[row, t] = 255
    return img[::-1].copy()   # flip: high price → top row


# ===========================================================================
# CNN MODEL — exact copy of baseline.py
# Only change: flatten size computed dynamically (so it works for all 3 scales)
# Original hardcodes 46080 which only works for the 20d (64×60) model.
# ===========================================================================

class Net(nn.Module):
    """
    Exact replica of Stock_CNN/models/baseline.py.
    Three blocks: Conv2d → BatchNorm2d → LeakyReLU(0.01) → MaxPool2d
    followed by Dropout(0.5) → Linear(flat→2).
    """
    def __init__(self, image_height, image_width):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64,  kernel_size=(5,3), stride=(3,1),
                      dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2,1), stride=(2,1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5,3), stride=(3,1),
                      dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2,1), stride=(2,1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5,3), stride=(3,1),
                      dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2,1), stride=(2,1)),
        )

        # Dynamic flatten size — same as paper but works for all three scales
        with torch.no_grad():
            d = torch.zeros(1, 1, image_height, image_width)
            flat = self.layer3(self.layer2(self.layer1(d))).view(1, -1).shape[1]

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(flat, 2),
        )
        self._flat = flat

        # Xavier uniform init — same as init_weights() in train.ipynb
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def features(self, x):
        """Pre-FC representation — for LightGBM feature extraction."""
        x = x.view(-1, 1, x.shape[-2], x.shape[-1])
        x = self.layer3(self.layer2(self.layer1(x)))
        return x.view(x.size(0), -1)

    def forward(self, x):
        return self.fc1(self.features(x))


# ===========================================================================
# TRAINING — exact train_loop / val_loop logic from train.ipynb
# ===========================================================================

def train_one_model(X_tr, y_tr, X_val, y_val, image_height, image_width, device):
    """
    Trains one Net instance following train.ipynb exactly:
      - CrossEntropyLoss() (no weighting)
      - Adam(lr=1e-5)
      - Early stopping patience=5, max 100 epochs
      - Batch size 128 train / 256 val
      - Xavier uniform init (already applied in Net.__init__)
    Returns trained model.
    """
    net     = Net(image_height, image_width).to(device)
    loss_fn = nn.CrossEntropyLoss()                        # original: no weighting
    opt     = optim.Adam(net.parameters(), lr=LR)          # lr=1e-5

    # Tensors — images are uint8 0/255, convert to float32 for the network
    Xt = torch.tensor(X_tr[:, None].astype(np.float32) / 255.0)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xv = torch.tensor(X_val[:, None].astype(np.float32) / 255.0)
    yv = torch.tensor(y_val, dtype=torch.long)
    N  = len(y_tr)

    min_val_loss  = 1e9
    last_min_ind  = -1
    best_state    = None

    for epoch in range(MAX_EPOCHS):
        # ── train loop (train.ipynb) ────────────────────────────────────────
        net.train()
        perm = torch.randperm(N)
        for start in range(0, N, BATCH_TRAIN):
            idx = perm[start: start + BATCH_TRAIN]
            xb  = Xt[idx].to(device)
            yb  = yt[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()

        # ── val loop (train.ipynb) ──────────────────────────────────────────
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(yv), BATCH_VAL):
                xb = Xv[start: start + BATCH_VAL].to(device)
                yb = yv[start: start + BATCH_VAL].to(device)
                val_loss += loss_fn(net(xb), yb).item()

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            last_min_ind = epoch
            best_state   = {k: v.cpu().clone()
                            for k, v in net.state_dict().items()}
        elif epoch - last_min_ind >= EARLY_STOP:
            break

    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return net


def get_logits(net, X, device):
    """
    Returns Softmax probability of class 1 (UP) for all samples.
    Matches test.ipynb:  predict_logit = Softmax(y_pred)[:,1]
    """
    net.eval()
    Xt    = torch.tensor(X[:, None].astype(np.float32) / 255.0)
    probs = []
    with torch.no_grad():
        for start in range(0, len(X), BATCH_VAL):
            xb = Xt[start: start + BATCH_VAL].to(device)
            p  = torch.softmax(net(xb), dim=1)[:, 1]
            probs.append(p.cpu().numpy())
    return np.hstack(probs)


def get_features(net, X, device):
    """Pre-FC features for LightGBM."""
    net.eval()
    Xt   = torch.tensor(X[:, None].astype(np.float32) / 255.0)
    feat = []
    with torch.no_grad():
        for start in range(0, len(X), BATCH_VAL):
            xb = Xt[start: start + BATCH_VAL].to(device)
            feat.append(net.features(xb).cpu().numpy())
    return np.vstack(feat)


# ===========================================================================
# DATA — build price images for all window sizes
# ===========================================================================

def build_images(price_arr, indices, windows, image_sizes):
    """Build price images for a set of sample indices."""
    out = {}
    for w in windows:
        H, W = image_sizes[w]
        imgs = np.zeros((len(indices), H, W), dtype=np.uint8)
        for s, gi in enumerate(indices):
            imgs[s] = price_to_image(price_arr[gi - w: gi], H, W)
        out[w] = imgs
    return out


# ===========================================================================
# STEP 1 — Load data
# ===========================================================================
print("\n[1/4] Loading data...")

data_raw       = econ_df_1[FEATURES].copy()
data_raw.index = pd.to_datetime(data_raw.index)
data_raw       = (data_raw
                  .reindex(pd.date_range(data_raw.index.min(),
                                         data_raw.index.max(), freq='D'))
                  .ffill().dropna())

global_wti = data_raw['WTI'].values.copy()
N_data     = len(data_raw)
SPLIT      = pd.Timestamp('2020-01-01')

print(f"  {data_raw.index.min().date()} → {data_raw.index.max().date()}")
print(f"  Days: {N_data:,}")


# ===========================================================================
# STEP 2 — Labels
# ===========================================================================
print("\n[2/4] Computing labels...")

ret5   = (pd.Series(global_wti).shift(-HORIZON) -
          pd.Series(global_wti)) / pd.Series(global_wti)
labels = np.where(ret5.values >  THRESHOLD,  1,
         np.where(ret5.values < -THRESHOLD,  0, -1))

max_w  = max(WINDOWS)
s_idx, s_lbl, s_dates = [], [], []
for i in range(max_w, N_data - HORIZON):
    if labels[i] == -1:
        continue
    s_idx.append(i); s_lbl.append(int(labels[i]))
    s_dates.append(data_raw.index[i])

s_idx   = np.array(s_idx)
s_lbl   = np.array(s_lbl)
s_dates = pd.DatetimeIndex(s_dates)
N_S     = len(s_lbl)
bah_acc = (s_lbl == 1).mean() * 100

print(f"  Samples: {N_S:,}  Up: {(s_lbl==1).sum():,}  Down: {(s_lbl==0).sum():,}")
print(f"  Buy & hold: {bah_acc:.2f}%")


# ===========================================================================
# STEP 3 — Walk-forward backtest
# ===========================================================================
print("\n[3/4] Walk-forward backtest...")

init_tr = int(N_S * INIT_TRAIN)
n_folds = (N_S - init_tr + STEP_SIZE - 1) // STEP_SIZE
print(f"  Folds: {n_folds}  Init train: {init_tr:,}  Step: {STEP_SIZE}")
print(f"  CNN: Adam lr={LR}, max_epochs={MAX_EPOCHS}, patience={EARLY_STOP}")
print(f"  Confidence threshold: {CONF_THRESHOLD} (from test.ipynb)")

all_true, all_pred, all_mask = [], [], []
all_dates_out, all_returns   = [], []
fold_info                    = []
next_trade                   = 0
cursor                       = init_tr

with tqdm(total=n_folds, desc="  Folds") as pbar:
    while cursor < N_S:
        te_end = min(cursor + STEP_SIZE, N_S)
        idx_tr = s_idx[:cursor];     y_tr = s_lbl[:cursor]
        idx_te = s_idx[cursor:te_end]; y_te = s_lbl[cursor:te_end]

        if len(y_tr) < MIN_TRAIN:
            cursor += STEP_SIZE; pbar.update(1); continue

        # ── Fold-level WTI price scaling ────────────────────────────────────
        # Scale only WTI (used for image encoding) on training rows
        tr_rows = np.unique([gi - w for gi in idx_tr for w in WINDOWS])
        tr_rows = tr_rows[tr_rows >= 0]
        sc      = MinMaxScaler()
        sc.fit(global_wti[tr_rows].reshape(-1, 1))
        wti_sc  = sc.transform(global_wti.reshape(-1, 1)).ravel()

        # Build images using scaled WTI price
        X_tr_im = build_images(wti_sc, idx_tr, WINDOWS, IMAGE_SIZE)
        X_te_im = build_images(wti_sc, idx_te, WINDOWS, IMAGE_SIZE)

        # ── Train/val split — 70/30 sequential (same as original) ───────────
        split_idx  = int(len(y_tr) * TRAIN_RATIO)
        X_tr2  = {w: X_tr_im[w][:split_idx] for w in WINDOWS}
        X_val  = {w: X_tr_im[w][split_idx:]  for w in WINDOWS}
        y_tr2  = y_tr[:split_idx]
        y_val  = y_tr[split_idx:]

        try:
            # ── Train one Net per window (same as original) ──────────────────
            nets = {}
            for w in WINDOWS:
                H, W  = IMAGE_SIZE[w]
                nets[w] = train_one_model(
                    X_tr2[w], y_tr2, X_val[w], y_val, H, W, DEVICE)

            # ── Get Softmax logits for test set (same as test.ipynb) ─────────
            # predict_logit = Softmax(y_pred)[:,1]
            logits = {w: get_logits(nets[w], X_te_im[w], DEVICE)
                      for w in WINDOWS}

            # ── Apply confidence threshold 0.58 (same as test.ipynb) ────────
            # Original: label_filtered = label_df[predict_logit > threshold]
            # For multi-scale: use mean logit across the three branches
            mean_logit = np.mean([logits[w] for w in WINDOWS], axis=0)
            preds      = (mean_logit >= 0.5).astype(int)     # direction
            conf_mask  = mean_logit > CONF_THRESHOLD          # 0.58 filter

            # ── Additionally: extract CNN features → LightGBM ───────────────
            # (professor's multi-scale concatenation extension)
            feat_tr = np.hstack([get_features(nets[w], X_tr_im[w], DEVICE)
                                  for w in WINDOWS])
            feat_te = np.hstack([get_features(nets[w], X_te_im[w], DEVICE)
                                  for w in WINDOWS])
            imp     = SimpleImputer(strategy='mean')
            feat_tr = imp.fit_transform(feat_tr)
            feat_te = imp.transform(feat_te)

            clf = lgb.LGBMClassifier(
                objective='binary', verbosity=-1,
                n_estimators=200, learning_rate=0.05,
                num_leaves=31, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1)
            clf.fit(feat_tr[:split_idx], y_tr2,
                    eval_set=[(feat_tr[split_idx:], y_val)],
                    callbacks=[lgb.early_stopping(20, verbose=False),
                               lgb.log_evaluation(-1)])

            lgb_proba  = clf.predict_proba(feat_te)[:, 1]
            lgb_preds  = (lgb_proba >= 0.5).astype(int)
            lgb_conf   = lgb_proba > CONF_THRESHOLD           # same 0.58 threshold

            # Final mask: CNN confidence AND LightGBM confidence both pass 0.58
            final_mask = conf_mask & lgb_conf
            preds      = lgb_preds   # use LightGBM prediction direction

            # ── Non-overlapping trades ───────────────────────────────────────
            no_mask = np.zeros(len(y_te), dtype=bool)
            last_g  = next_trade
            for i in range(len(y_te)):
                gi = int(idx_te[i])
                if final_mask[i] and gi >= last_g:
                    no_mask[i] = True
                    last_g     = gi + HORIZON
            next_trade = last_g
            final_mask = final_mask & no_mask

            # ── Returns ──────────────────────────────────────────────────────
            fold_ret = np.zeros(len(y_te))
            for i in range(len(y_te)):
                gi  = int(idx_te[i])
                p0  = global_wti[gi]
                p1  = global_wti[gi+HORIZON] if gi+HORIZON < N_data else p0
                ret = (p1 - p0) / (p0 + 1e-8)
                fold_ret[i] = ret if preds[i] == 1 else -ret

        except Exception as e:
            print(f"\n  Fold error: {e}")
            preds = np.full(len(y_te), -1)
            final_mask = np.zeros(len(y_te), dtype=bool)
            fold_ret   = np.zeros(len(y_te))

        fd = s_dates[cursor] if cursor < len(s_dates) else s_dates[-1]
        fa = (accuracy_score(y_te[final_mask], preds[final_mask])
              if final_mask.sum() > 0 else 0.0)
        fold_info.append({'fold': len(fold_info), 'acc': fa,
                          'period': 'post' if fd >= SPLIT else 'pre'})
        all_true.extend(y_te.tolist())
        all_pred.extend(preds.tolist())
        all_mask.extend(final_mask.tolist())
        all_dates_out.extend(s_dates[cursor:te_end].tolist())
        all_returns.extend(fold_ret.tolist())
        cursor += STEP_SIZE; pbar.update(1)

all_true    = np.array(all_true);    all_pred  = np.array(all_pred)
all_mask    = np.array(all_mask);    all_dates = pd.DatetimeIndex(all_dates_out)
all_returns = np.array(all_returns); fold_df   = pd.DataFrame(fold_info)


# ===========================================================================
# STEP 4 — Results
# ===========================================================================
print("\n" + "="*70)
print("  RESULTS  |  Multi-Scale CNN + LightGBM  |  WTI Price Only")
print("="*70)

vm   = all_mask
acc  = accuracy_score(all_true[vm], all_pred[vm])
f1   = f1_score(all_true[vm], all_pred[vm], average='macro', zero_division=0)
dr   = (all_pred[vm&(all_true==0)]==0).mean() if (vm&(all_true==0)).any() else 0
ur   = (all_pred[vm&(all_true==1)]==1).mean() if (vm&(all_true==1)).any() else 0

print(f"\n  Predicted : {vm.sum():,}/{len(all_true):,}  Acc: {acc*100:.2f}%"
      f"  F1: {f1:.4f}  Down Rec: {dr:.3f}  Up Rec: {ur:.3f}")
print(f"\n{classification_report(all_true[vm], all_pred[vm], target_names=['Down','Up'], digits=4)}")

spm = np.array([d >= SPLIT for d in all_dates])
print("  ── Period Breakdown ──────────────────────────────────────────")
for lbl, pm in [("Pre-2020", ~spm&vm), ("Post-2020", spm&vm)]:
    if pm.sum() > 5:
        a = accuracy_score(all_true[pm], all_pred[pm])*100
        f = f1_score(all_true[pm], all_pred[pm], average='macro', zero_division=0)
        print(f"  {lbl}: Acc={a:.2f}%  F1={f:.4f}  Trades={pm.sum()}")

tr   = all_returns[vm] - 2*TRADE_FEE
sh   = (tr.mean()/(tr.std()+1e-8)) * np.sqrt(252/HORIZON)
tot  = tr.sum()*100
mdd  = (np.maximum.accumulate(np.cumsum(tr)) - np.cumsum(tr)).max()*100
print(f"\n  Total return: {tot:+.2f}%  Sharpe: {sh:.3f}  Max DD: {mdd:.2f}%")
print(f"  Baselines — Random: 50.00%  Buy&hold: {bah_acc:.2f}%")

# Chart
fig, ax = plt.subplots(figsize=(14, 4), facecolor='white')
ax.set_facecolor('white')
fa   = fold_df['acc'].values*100
cols = ['#A371F7' if r['period']=='post' else '#1A5276'
        for _, r in fold_df.iterrows()]
ax.bar(range(len(fa)), fa, color=cols, alpha=0.85, width=0.7)
ax.axhline(50, color='grey', ls='--', lw=0.9, label='50% random')
ax.axhline(acc*100, color='#C0392B', ls=':', lw=1.2,
           label=f'Overall {acc*100:.1f}%')
if 'post' in fold_df['period'].values:
    ps = fold_df[fold_df['period']=='post'].index.min()
    ax.axvspan(ps-0.5, len(fa)-0.5, alpha=0.08, color='#A371F7')
ax.set_xlabel('Fold'); ax.set_ylabel('Accuracy (%)')
ax.set_title('Multi-Scale CNN + LightGBM  |  WTI  |  '
             'Jiang, Kelly & Xiu (2023)\n'
             'Blue = Pre-2020 (price only)  |  Purple = Post-2020',
             fontsize=10, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.4, axis='y')
plt.tight_layout()
plt.savefig('wti_cnn_backtest.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
print("\n  Chart saved → wti_cnn_backtest.png")
print("  Done.")

# Test on just the last 5 folds (post-2020 only)
# Takes ~15 minutes
# If accuracy is near 50% on those 5 folds → CNN is not working
# If accuracy is above 54% on some folds → worth running the full backtest
INIT_TRAIN = 0.92   # start from 92% — only test last 5 folds

"""
==============================================================================
  Multi-Scale CNN  |  WTI Crude Oil  |  Walk-Forward Backtest
  Jiang, Kelly & Xiu (2023) — "(Re-)Imag(in)ing Price Trends"
  github.com/lich99/Stock_CNN

  Architecture: three parallel CNN branches (5d / 20d / 60d)
    → features concatenated
    → single shared classification head (Linear → 2)
    → end-to-end, no LightGBM

  QUICK SANITY CHECK MODE (QUICK_CHECK = True):
    Starts at 92% of data → tests only last ~5 folds (~15 minutes)
    If those folds show signal → run full backtest (QUICK_CHECK = False)

  Matches original paper:
    · CNN architecture: Conv2d(1→64→128→256, kernel=(5,3), dilation=(2,1),
                        stride=(3,1), padding=(12,1)) + BN + LeakyReLU(0.01)
                        + MaxPool2d(2,1)
    · Loss:        CrossEntropyLoss() — no class weighting
    · Optimizer:   Adam(lr=1e-5)
    · Early stop:  patience=5, max=100 epochs
    · Batch:       128 train / 256 val
    · Split:       70/30 sequential
    · Threshold:   0.58 confidence
    · Images:      1 pixel per price level, uint8
    · Image sizes: 5d→(32,15)  20d→(64,60)  60d→(96,180)

  WTI adaptations (unavoidable):
    · Walk-forward backtest (one asset, not thousands of stocks)
    · ±0.3% label threshold (removes near-zero noise moves)
    · Multi-scale concat → shared head (professor's extension)
==============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("  Multi-Scale CNN  |  WTI  |  Jiang, Kelly & Xiu (2023)")
print("=" * 70)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

# ── Quick sanity check ──────────────────────────────────────────────────────
# Set True first: tests only the last ~5 folds (~15 min on Colab CPU)
# If results look promising, set False to run the full 72-fold backtest
QUICK_CHECK = True    # ← change to False for full backtest

# ── Image sizes — exact from train.ipynb ────────────────────────────────────
IMAGE_SIZE = {5: (32, 15), 20: (64, 60), 60: (96, 180)}
WINDOWS    = [5, 20, 60]

# ── Training — exact from original paper ────────────────────────────────────
LR          = 1e-5       # Adam lr=1e-5
MAX_EPOCHS  = 100        # original: 100
EARLY_STOP  = 5          # original: patience=5
BATCH_TRAIN = 128        # original: 128
BATCH_VAL   = 256        # original: 256
TRAIN_RATIO = 0.70       # original: 70/30 split

# ── Prediction threshold — from test.ipynb ──────────────────────────────────
CONF_THRESHOLD = 0.58    # original threshold

# ── WTI settings ────────────────────────────────────────────────────────────
FEATURES  = ['WTI', 'Brent', 'Gold', 'Dollar/Euro', 'Copper', 'S&P500']
HORIZON   = 5
THRESHOLD = 0.003

# ── Backtest ─────────────────────────────────────────────────────────────────
STEP_SIZE  = 60
MIN_TRAIN  = 300
TRADE_FEE  = 0.001

# Quick check starts at 92% → only ~5 folds
INIT_TRAIN = 0.92 if QUICK_CHECK else 0.50

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n  QUICK_CHECK = {QUICK_CHECK}  "
      f"({'last ~5 folds only' if QUICK_CHECK else 'full 72-fold backtest'})")
print(f"  Device: {DEVICE}  |  lr={LR}  epochs={MAX_EPOCHS}  "
      f"patience={EARLY_STOP}  threshold={CONF_THRESHOLD}")


# ===========================================================================
# PRICE → 2D IMAGE  (matches original: 1 pixel, uint8 0/255)
# ===========================================================================

def price_to_image(price_window, height, width):
    """
    Converts 1D price series (len=width) to 2D uint8 image (H×W).
    One pixel per time step at the discretised price level.
    Pixel = 255 (white), background = 0 (black).
    High price → top row (chart convention).
    """
    p_min = price_window.min()
    p_max = price_window.max()
    p_norm = (price_window - p_min) / (p_max - p_min + 1e-8)
    img = np.zeros((height, width), dtype=np.uint8)
    for t, pn in enumerate(p_norm):
        row = int(np.clip(pn * (height - 1), 0, height - 1))
        img[row, t] = 255
    return img[::-1].copy()   # flip: high price = top


def build_images(price_arr, indices, windows, image_sizes):
    """Build price images for all window sizes."""
    out = {}
    for w in windows:
        H, W  = image_sizes[w]
        imgs  = np.zeros((len(indices), H, W), dtype=np.uint8)
        for s, gi in enumerate(indices):
            imgs[s] = price_to_image(price_arr[gi - w: gi], H, W)
        out[w] = imgs
    return out


# ===========================================================================
# CNN BRANCH  (exact baseline.py architecture)
# ===========================================================================

class CNNBranch(nn.Module):
    """
    Single CNN branch — exact copy of Stock_CNN/models/baseline.py.
    Returns flat feature vector (before classification head).
    """
    def __init__(self, image_height, image_width):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64,  kernel_size=(5,3), stride=(3,1),
                      dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2,1), stride=(2,1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5,3), stride=(3,1),
                      dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2,1), stride=(2,1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5,3), stride=(3,1),
                      dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2,1), stride=(2,1)),
        )
        # Dynamic flatten size — same logic as paper, works for all 3 scales
        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_height, image_width)
            out   = self.layer3(self.layer2(self.layer1(dummy)))
            self.flat_size = out.view(1, -1).shape[1]

    def forward(self, x):
        x = x.view(-1, 1, x.shape[-2], x.shape[-1])
        x = self.layer3(self.layer2(self.layer1(x)))
        return x.view(x.size(0), -1)   # (batch, flat_size)


# ===========================================================================
# MULTI-SCALE MODEL  (professor's extension — end-to-end, no LightGBM)
# ===========================================================================

class MultiScaleCNN(nn.Module):
    """
    Three parallel CNN branches (5d / 20d / 60d).
    Features concatenated → Dropout(0.5) → Linear(→2).
    Trained end-to-end with CrossEntropyLoss.

    forward() takes a dict: {window_len: (batch, H, W) tensor}
    """
    def __init__(self, windows, image_sizes):
        super().__init__()
        self.windows  = windows
        self.branches = nn.ModuleDict({
            str(w): CNNBranch(*image_sizes[w]) for w in windows
        })
        total = sum(self.branches[str(w)].flat_size for w in windows)
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),           # same as original fc1
            nn.Linear(total, 2),
        )
        # Xavier uniform init — same as init_weights() in train.ipynb
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x_dict):
        parts = [self.branches[str(w)](x_dict[w]) for w in self.windows]
        return self.head(torch.cat(parts, dim=1))


# ===========================================================================
# TRAINING  (follows train.ipynb exactly)
# ===========================================================================

def to_tensor(imgs_dict, windows, device=None):
    """Convert dict of uint8 numpy images to float32 tensors in [0,1]."""
    out = {}
    for w in windows:
        t = torch.tensor(
            imgs_dict[w].astype(np.float32) / 255.0
        ).unsqueeze(1)   # (N, 1, H, W)  — but we squeeze channel in branch.forward
        out[w] = t if device is None else t.to(device)
    return out


def train_model(X_tr_imgs, y_tr, X_val_imgs, y_val, windows, image_sizes, device):
    """
    Trains MultiScaleCNN following train.ipynb:
      - CrossEntropyLoss (no weighting)
      - Adam(lr=1e-5)
      - Early stopping patience=5, max 100 epochs
      - Batch 128 train / 256 val
    Returns trained model.
    """
    model   = MultiScaleCNN(windows, image_sizes).to(device)
    loss_fn = nn.CrossEntropyLoss()          # original: no weighting
    opt     = optim.Adam(model.parameters(), lr=LR)

    # Pre-load to CPU tensors
    Xt = {w: torch.tensor(X_tr_imgs[w].astype(np.float32)/255.0) for w in windows}
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xv = {w: torch.tensor(X_val_imgs[w].astype(np.float32)/255.0) for w in windows}
    yv = torch.tensor(y_val, dtype=torch.long)
    N  = len(y_tr)

    best_val  = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        # ── train_loop (train.ipynb) ─────────────────────────────────────────
        model.train()
        perm = torch.randperm(N)
        for start in range(0, N, BATCH_TRAIN):
            idx = perm[start: start + BATCH_TRAIN]
            xb  = {w: Xt[w][idx].to(device) for w in windows}
            yb  = yt[idx].to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        # ── val_loop (train.ipynb) ───────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(yv), BATCH_VAL):
                xb = {w: Xv[w][start:start+BATCH_VAL].to(device)
                      for w in windows}
                yb = yv[start:start+BATCH_VAL].to(device)
                val_loss += loss_fn(model(xb), yb).item()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def predict(model, X_imgs, windows, device):
    """
    Returns Softmax probability of class 1 (UP).
    Matches test.ipynb: predict_logit = Softmax(y_pred)[:,1]
    """
    model.eval()
    Xt   = {w: torch.tensor(X_imgs[w].astype(np.float32)/255.0)
            for w in windows}
    N    = len(next(iter(Xt.values())))
    prob = []
    with torch.no_grad():
        for start in range(0, N, BATCH_VAL):
            xb = {w: Xt[w][start:start+BATCH_VAL].to(device)
                  for w in windows}
            p  = torch.softmax(model(xb), dim=1)[:, 1]
            prob.append(p.cpu().numpy())
    return np.hstack(prob)


# ===========================================================================
# STEP 1 — Data
# ===========================================================================
print("\n[1/4] Loading data...")

data_raw       = econ_df_1[FEATURES].copy()
data_raw.index = pd.to_datetime(data_raw.index)
data_raw       = (data_raw
                  .reindex(pd.date_range(data_raw.index.min(),
                                         data_raw.index.max(), freq='D'))
                  .ffill().dropna())

global_wti = data_raw['WTI'].values.copy()
N_data     = len(data_raw)
SPLIT      = pd.Timestamp('2020-01-01')

print(f"  {data_raw.index.min().date()} → {data_raw.index.max().date()}"
      f"  ({N_data:,} days)")


# ===========================================================================
# STEP 2 — Labels
# ===========================================================================
print("\n[2/4] Computing labels...")

ret5   = (pd.Series(global_wti).shift(-HORIZON) -
          pd.Series(global_wti)) / pd.Series(global_wti)
raw_lbl = np.where(ret5.values >  THRESHOLD,  1,
          np.where(ret5.values < -THRESHOLD,  0, -1))

max_w  = max(WINDOWS)
s_idx, s_lbl, s_dates = [], [], []
for i in range(max_w, N_data - HORIZON):
    if raw_lbl[i] == -1:
        continue
    s_idx.append(i); s_lbl.append(int(raw_lbl[i]))
    s_dates.append(data_raw.index[i])

s_idx   = np.array(s_idx)
s_lbl   = np.array(s_lbl)
s_dates = pd.DatetimeIndex(s_dates)
N_S     = len(s_lbl)
bah_acc = (s_lbl == 1).mean() * 100

print(f"  Samples: {N_S:,}  "
      f"Up: {(s_lbl==1).sum():,}  Down: {(s_lbl==0).sum():,}")
print(f"  Buy & hold baseline: {bah_acc:.2f}%")


# ===========================================================================
# STEP 3 — Walk-forward backtest
# ===========================================================================
print("\n[3/4] Walk-forward backtest...")

init_tr = int(N_S * INIT_TRAIN)
n_folds = (N_S - init_tr + STEP_SIZE - 1) // STEP_SIZE
est_min = n_folds * 3   # ~3 min per fold on Colab CPU

print(f"  Init train: {init_tr:,} samples  Step: {STEP_SIZE}")
print(f"  Folds: {n_folds}  Estimated time: ~{est_min} min")
if QUICK_CHECK:
    print(f"  QUICK CHECK — testing last {n_folds} folds only")
    print(f"  If results look good → set QUICK_CHECK=False for full run")

all_true, all_pred, all_mask = [], [], []
all_dates_out, all_returns   = [], []
fold_info                    = []
next_trade                   = 0
cursor                       = init_tr

with tqdm(total=n_folds, desc="  Folds") as pbar:
    while cursor < N_S:
        te_end = min(cursor + STEP_SIZE, N_S)
        idx_tr = s_idx[:cursor];       y_tr = s_lbl[:cursor]
        idx_te = s_idx[cursor:te_end]; y_te = s_lbl[cursor:te_end]

        if len(y_tr) < MIN_TRAIN:
            cursor += STEP_SIZE; pbar.update(1); continue

        # ── Scale WTI on training rows only ──────────────────────────────────
        tr_rows = np.unique([gi - w for gi in idx_tr for w in WINDOWS])
        tr_rows = tr_rows[tr_rows >= 0]
        sc      = MinMaxScaler()
        sc.fit(global_wti[tr_rows].reshape(-1, 1))
        wti_sc  = sc.transform(global_wti.reshape(-1, 1)).ravel()

        # ── Build images ──────────────────────────────────────────────────────
        X_tr_im = build_images(wti_sc, idx_tr, WINDOWS, IMAGE_SIZE)
        X_te_im = build_images(wti_sc, idx_te, WINDOWS, IMAGE_SIZE)

        # ── 70/30 sequential split (original) ────────────────────────────────
        split   = int(len(y_tr) * TRAIN_RATIO)
        X_tr2   = {w: X_tr_im[w][:split] for w in WINDOWS}
        X_val   = {w: X_tr_im[w][split:]  for w in WINDOWS}
        y_tr2   = y_tr[:split]
        y_val   = y_tr[split:]

        try:
            # ── Train end-to-end multi-scale CNN ─────────────────────────────
            model = train_model(X_tr2, y_tr2, X_val, y_val,
                                WINDOWS, IMAGE_SIZE, DEVICE)

            # ── Predict: Softmax[:,1] then threshold 0.58 (test.ipynb) ───────
            prob       = predict(model, X_te_im, WINDOWS, DEVICE)
            preds      = (prob >= 0.5).astype(int)
            final_mask = prob > CONF_THRESHOLD    # original: predict_logit > 0.58

            # ── Non-overlapping trades ────────────────────────────────────────
            no_m   = np.zeros(len(y_te), dtype=bool)
            last_g = next_trade
            for i in range(len(y_te)):
                gi = int(idx_te[i])
                if final_mask[i] and gi >= last_g:
                    no_m[i] = True
                    last_g  = gi + HORIZON
            next_trade = last_g
            final_mask = final_mask & no_m

            # ── Returns ───────────────────────────────────────────────────────
            fold_ret = np.zeros(len(y_te))
            for i in range(len(y_te)):
                gi  = int(idx_te[i])
                p0  = global_wti[gi]
                p1  = global_wti[gi+HORIZON] if gi+HORIZON < N_data else p0
                ret = (p1 - p0) / (p0 + 1e-8)
                fold_ret[i] = ret if preds[i] == 1 else -ret

        except Exception as e:
            print(f"\n  Fold {len(fold_info)} error: {e}")
            preds      = np.full(len(y_te), -1)
            final_mask = np.zeros(len(y_te), dtype=bool)
            fold_ret   = np.zeros(len(y_te))

        fd = s_dates[cursor] if cursor < len(s_dates) else s_dates[-1]
        fa = (accuracy_score(y_te[final_mask], preds[final_mask])
              if final_mask.sum() > 0 else 0.0)
        fold_info.append({
            'fold'  : len(fold_info),
            'acc'   : fa,
            'n_pred': int(final_mask.sum()),
            'period': 'post' if fd >= SPLIT else 'pre',
            'date'  : fd,
        })

        all_true.extend(y_te.tolist())
        all_pred.extend(preds.tolist())
        all_mask.extend(final_mask.tolist())
        all_dates_out.extend(s_dates[cursor:te_end].tolist())
        all_returns.extend(fold_ret.tolist())
        cursor += STEP_SIZE; pbar.update(1)

all_true    = np.array(all_true);    all_pred  = np.array(all_pred)
all_mask    = np.array(all_mask);    all_dates = pd.DatetimeIndex(all_dates_out)
all_returns = np.array(all_returns); fold_df   = pd.DataFrame(fold_info)


# ===========================================================================
# STEP 4 — Results
# ===========================================================================
print("\n" + "="*70)
mode_str = "QUICK CHECK (last ~5 folds)" if QUICK_CHECK else "FULL BACKTEST"
print(f"  RESULTS  |  Multi-Scale CNN  |  WTI  |  {mode_str}")
print("="*70)

vm = all_mask

if vm.sum() == 0:
    print("\n  No trades passed the confidence threshold.")
    print("  Try reducing CONF_THRESHOLD from 0.58 to 0.50")
else:
    acc = accuracy_score(all_true[vm], all_pred[vm])
    f1  = f1_score(all_true[vm], all_pred[vm], average='macro', zero_division=0)
    dr  = (all_pred[vm&(all_true==0)]==0).mean() if (vm&(all_true==0)).any() else 0
    ur  = (all_pred[vm&(all_true==1)]==1).mean() if (vm&(all_true==1)).any() else 0

    print(f"\n  Predicted : {vm.sum():,} / {len(all_true):,} "
          f"({vm.mean()*100:.1f}%)")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"  Down Rec  : {dr:.3f}    Up Rec : {ur:.3f}")
    print(f"\n{classification_report(all_true[vm], all_pred[vm], target_names=['Down','Up'], digits=4)}")

    spm = np.array([d >= SPLIT for d in all_dates])
    print("  ── Period Breakdown ────────────────────────────────────────────")
    for lbl, pm in [("Pre-2020", ~spm&vm), ("Post-2020", spm&vm)]:
        if pm.sum() > 5:
            a = accuracy_score(all_true[pm], all_pred[pm])*100
            f = f1_score(all_true[pm], all_pred[pm], average='macro', zero_division=0)
            print(f"  {lbl}: Acc={a:.2f}%  F1={f:.4f}  Trades={pm.sum()}")

    tr  = all_returns[vm] - 2*TRADE_FEE
    sh  = (tr.mean()/(tr.std()+1e-8)) * np.sqrt(252/HORIZON)
    tot = tr.sum()*100
    mdd = (np.maximum.accumulate(np.cumsum(tr)) - np.cumsum(tr)).max()*100
    print(f"\n  Total return : {tot:+.2f}%")
    print(f"  Sharpe ratio : {sh:.3f}")
    print(f"  Max drawdown : {mdd:.2f}%")
    print(f"\n  Baselines — Random: 50.00%  |  Buy&hold: {bah_acc:.2f}%")

    # ── Per-fold summary ──────────────────────────────────────────────────
    print(f"\n  ── Fold accuracy ────────────────────────────────────────────")
    print(f"  Mean : {fold_df['acc'].mean()*100:.2f}%  "
          f"Std : {fold_df['acc'].std()*100:.2f}%  "
          f"Min : {fold_df['acc'].min()*100:.2f}%  "
          f"Max : {fold_df['acc'].max()*100:.2f}%")
    for _, row in fold_df.iterrows():
        bar  = '█' * int(row['acc'] * 20)
        flag = ' ← POST-2020' if row['period'] == 'post' else ''
        print(f"  Fold {int(row['fold']):2d} "
              f"[{str(row['date'])[:10]}]  "
              f"{row['acc']*100:5.1f}%  {bar}{flag}")

    # ── Decision helper ───────────────────────────────────────────────────
    if QUICK_CHECK:
        mean_acc = fold_df['acc'].mean()
        print(f"\n  {'='*50}")
        if mean_acc >= 0.54:
            print(f"  ✅ Mean accuracy {mean_acc*100:.1f}% ≥ 54%")
            print(f"  Signal looks promising.")
            print(f"  → Set QUICK_CHECK = False to run full 72-fold backtest")
        elif mean_acc >= 0.51:
            print(f"  ⚠️  Mean accuracy {mean_acc*100:.1f}% — marginal signal")
            print(f"  → Consider transfer learning or smaller model before full run")
        else:
            print(f"  ❌ Mean accuracy {mean_acc*100:.1f}% — no signal detected")
            print(f"  → CNN is overfitting on small data")
            print(f"  → Recommendation: use transfer learning from pretrained weights")
            print(f"  → Pretrained weights available at:")
            print(f"     Stock_CNN-main/pt/baseline_epoch_5_train_0.704243_val_0.695599.pt")
        print(f"  {'='*50}")

    # ── Chart ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor='white')

    # Fold accuracy bars
    ax1 = axes[0]; ax1.set_facecolor('white')
    fa   = fold_df['acc'].values * 100
    cols = ['#A371F7' if r['period']=='post' else '#1A5276'
            for _, r in fold_df.iterrows()]
    ax1.bar(range(len(fa)), fa, color=cols, alpha=0.85, width=0.7)
    ax1.axhline(50,       color='grey', ls='--', lw=0.9, label='50% random')
    ax1.axhline(bah_acc,  color='orange', ls=':', lw=0.9,
                label=f'Buy&hold {bah_acc:.1f}%')
    ax1.axhline(acc*100,  color='red',  ls=':',  lw=1.2,
                label=f'Overall {acc*100:.1f}%')
    ax1.set_xlabel('Fold'); ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'Fold Accuracy — {mode_str}\n'
                  f'Blue=Pre-2020  Purple=Post-2020', fontweight='bold')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.4, axis='y')

    # Cumulative return
    ax2 = axes[1]; ax2.set_facecolor('white')
    cum = np.cumsum(all_returns[vm] - 2*TRADE_FEE) * 100
    ax2.plot(cum, color='#1A5276', linewidth=1.2)
    ax2.fill_between(range(len(cum)), 0, cum,
                     where=(cum >= 0), color='#1E8449', alpha=0.2)
    ax2.fill_between(range(len(cum)), 0, cum,
                     where=(cum < 0),  color='#C0392B', alpha=0.2)
    ax2.axhline(0, color='grey', lw=0.8, ls='--')
    ax2.set_xlabel('Trade number'); ax2.set_ylabel('Cumulative return (%)')
    ax2.set_title(f'Cumulative Return  |  Sharpe={sh:.2f}',
                  fontweight='bold')
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    fname = 'wti_cnn_quick.png' if QUICK_CHECK else 'wti_cnn_full.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"\n  Chart saved → {fname}")

print("\n  Done.")
if QUICK_CHECK:
    print("  Next step: check the decision above, then set QUICK_CHECK = False")

"""
==============================================================================
  Multi-Scale CNN  |  WTI Crude Oil  |  Walk-Forward Backtest
  Jiang, Kelly & Xiu (2023) — "(Re-)Imag(in)ing Price Trends"
  github.com/lich99/Stock_CNN

  Architecture: three parallel CNN branches (5d / 20d / 60d)
    → features concatenated
    → single shared classification head (Linear → 2)
    → end-to-end, no LightGBM

  QUICK SANITY CHECK MODE (QUICK_CHECK = True):
    Starts at 92% of data → tests only last ~5 folds (~15 minutes)
    If those folds show signal → run full backtest (QUICK_CHECK = False)

  Matches original paper:
    · CNN architecture: Conv2d(1→64→128→256, kernel=(5,3), dilation=(2,1),
                        stride=(3,1), padding=(12,1)) + BN + LeakyReLU(0.01)
                        + MaxPool2d(2,1)
    · Loss:        CrossEntropyLoss() — no class weighting
    · Optimizer:   Adam(lr=1e-5)
    · Early stop:  patience=5, max=100 epochs
    · Batch:       128 train / 256 val
    · Split:       70/30 sequential
    · Threshold:   0.58 confidence
    · Images:      1 pixel per price level, uint8
    · Image sizes: 5d→(32,15)  20d→(64,60)  60d→(96,180)

  WTI adaptations (unavoidable):
    · Walk-forward backtest (one asset, not thousands of stocks)
    · ±0.3% label threshold (removes near-zero noise moves)
    · Multi-scale concat → shared head (professor's extension)
==============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("  Multi-Scale CNN  |  WTI  |  Jiang, Kelly & Xiu (2023)")
print("=" * 70)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

# ── Quick sanity check ──────────────────────────────────────────────────────
# Set True first: tests only the last ~5 folds (~15 min on Colab CPU)
# If results look promising, set False to run the full 72-fold backtest
QUICK_CHECK = True    # ← change to False for full backtest

# ── Image sizes — exact from train.ipynb ────────────────────────────────────
IMAGE_SIZE = {5: (32, 15), 20: (64, 60), 60: (96, 180)}
WINDOWS    = [5, 20, 60]

# ── Training — exact from original paper ────────────────────────────────────
LR          = 1e-5       # Adam lr=1e-5
EARLY_STOP  = 5          # original: patience=5
BATCH_TRAIN = 128        # original: 128
BATCH_VAL   = 256        # original: 256
TRAIN_RATIO = 0.70       # original: 70/30 split

# ── Prediction threshold — from test.ipynb ──────────────────────────────────
CONF_THRESHOLD = 0.58    # original threshold

# ── WTI settings ────────────────────────────────────────────────────────────
FEATURES  = ['WTI', 'Brent', 'Gold', 'Dollar/Euro', 'Copper', 'S&P500']
HORIZON   = 5
THRESHOLD = 0.003

# ── Backtest ─────────────────────────────────────────────────────────────────
STEP_SIZE  = 60
MIN_TRAIN  = 300
TRADE_FEE  = 0.001

# Quick check starts at 92% → only ~5 folds
INIT_TRAIN = 0.92 if QUICK_CHECK else 0.50

# Reduce epochs in quick check mode — 15 is enough to detect signal
MAX_EPOCHS = 15 if QUICK_CHECK else 100

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n  QUICK_CHECK = {QUICK_CHECK}  "
      f"({'last ~5 folds only' if QUICK_CHECK else 'full 72-fold backtest'})")
print(f"  Device: {DEVICE}  |  lr={LR}  epochs={MAX_EPOCHS}  "
      f"patience={EARLY_STOP}  threshold={CONF_THRESHOLD}")


# ===========================================================================
# PRICE → 2D IMAGE  (matches original: 1 pixel, uint8 0/255)
# ===========================================================================

def price_to_image(price_window, height, width):
    """
    Converts 1D price series to 2D uint8 image (H×W).
    Vectorised — ~10x faster than pixel-by-pixel loop.

    The original paper uses 3 pixels per trading day (OHLC bar width),
    so image width = window_days × 3:
        5d  → width=15   (5  × 3)
        20d → width=60   (20 × 3)
        60d → width=180  (60 × 3)

    We interpolate the price series from len(price_window) to width
    so each day is spread across 3 pixels, matching the original.

    Pixel = 255 (white), background = 0 (black).
    High price → top row (chart convention).
    """
    # Interpolate price_window (n days) → width pixels
    x_old  = np.linspace(0, 1, len(price_window))
    x_new  = np.linspace(0, 1, width)
    p_interp = np.interp(x_new, x_old, price_window.astype(float))

    p_min  = p_interp.min()
    p_max  = p_interp.max()
    p_norm = (p_interp - p_min) / (p_max - p_min + 1e-8)
    rows   = np.clip((p_norm * (height - 1)).astype(int), 0, height - 1)
    img    = np.zeros((height, width), dtype=np.uint8)
    img[rows, np.arange(width)] = 255   # vectorised — one numpy op
    return img[::-1].copy()             # flip: high price = top


def build_images(price_arr, indices, windows, image_sizes):
    """Build price images for all window sizes."""
    out = {}
    for w in windows:
        H, W  = image_sizes[w]
        imgs  = np.zeros((len(indices), H, W), dtype=np.uint8)
        for s, gi in enumerate(indices):
            imgs[s] = price_to_image(price_arr[gi - w: gi], H, W)
        out[w] = imgs
    return out


# ===========================================================================
# CNN BRANCH  (exact baseline.py architecture)
# ===========================================================================

class CNNBranch(nn.Module):
    """
    Single CNN branch — exact copy of Stock_CNN/models/baseline.py.
    Returns flat feature vector (before classification head).
    """
    def __init__(self, image_height, image_width):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64,  kernel_size=(5,3), stride=(3,1),
                      dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2,1), stride=(2,1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5,3), stride=(3,1),
                      dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2,1), stride=(2,1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5,3), stride=(3,1),
                      dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2,1), stride=(2,1)),
        )
        # Dynamic flatten size — same logic as paper, works for all 3 scales
        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_height, image_width)
            out   = self.layer3(self.layer2(self.layer1(dummy)))
            self.flat_size = out.view(1, -1).shape[1]

    def forward(self, x):
        x = x.view(-1, 1, x.shape[-2], x.shape[-1])
        x = self.layer3(self.layer2(self.layer1(x)))
        return x.view(x.size(0), -1)   # (batch, flat_size)


# ===========================================================================
# MULTI-SCALE MODEL  (professor's extension — end-to-end, no LightGBM)
# ===========================================================================

class MultiScaleCNN(nn.Module):
    """
    Three parallel CNN branches (5d / 20d / 60d).
    Features concatenated → Dropout(0.5) → Linear(→2).
    Trained end-to-end with CrossEntropyLoss.

    forward() takes a dict: {window_len: (batch, H, W) tensor}
    """
    def __init__(self, windows, image_sizes):
        super().__init__()
        self.windows  = windows
        self.branches = nn.ModuleDict({
            str(w): CNNBranch(*image_sizes[w]) for w in windows
        })
        total = sum(self.branches[str(w)].flat_size for w in windows)
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),           # same as original fc1
            nn.Linear(total, 2),
        )
        # Xavier uniform init — same as init_weights() in train.ipynb
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x_dict):
        parts = [self.branches[str(w)](x_dict[w]) for w in self.windows]
        return self.head(torch.cat(parts, dim=1))


# ===========================================================================
# TRAINING  (follows train.ipynb exactly)
# ===========================================================================

def to_tensor(imgs_dict, windows, device=None):
    """Convert dict of uint8 numpy images to float32 tensors in [0,1]."""
    out = {}
    for w in windows:
        t = torch.tensor(
            imgs_dict[w].astype(np.float32) / 255.0
        ).unsqueeze(1)   # (N, 1, H, W)  — but we squeeze channel in branch.forward
        out[w] = t if device is None else t.to(device)
    return out


def train_model(X_tr_imgs, y_tr, X_val_imgs, y_val, windows, image_sizes, device):
    """
    Trains MultiScaleCNN following train.ipynb:
      - CrossEntropyLoss (no weighting)
      - Adam(lr=1e-5)
      - Early stopping patience=5, max 100 epochs
      - Batch 128 train / 256 val
    Returns trained model.
    """
    model   = MultiScaleCNN(windows, image_sizes).to(device)
    loss_fn = nn.CrossEntropyLoss()          # original: no weighting
    opt     = optim.Adam(model.parameters(), lr=LR)

    # Pre-load to CPU tensors
    Xt = {w: torch.tensor(X_tr_imgs[w].astype(np.float32)/255.0) for w in windows}
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xv = {w: torch.tensor(X_val_imgs[w].astype(np.float32)/255.0) for w in windows}
    yv = torch.tensor(y_val, dtype=torch.long)
    N  = len(y_tr)

    best_val  = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        # ── train_loop (train.ipynb) ─────────────────────────────────────────
        model.train()
        perm = torch.randperm(N)
        for start in range(0, N, BATCH_TRAIN):
            idx = perm[start: start + BATCH_TRAIN]
            xb  = {w: Xt[w][idx].to(device) for w in windows}
            yb  = yt[idx].to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        # ── val_loop (train.ipynb) ───────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(yv), BATCH_VAL):
                xb = {w: Xv[w][start:start+BATCH_VAL].to(device)
                      for w in windows}
                yb = yv[start:start+BATCH_VAL].to(device)
                val_loss += loss_fn(model(xb), yb).item()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                if QUICK_CHECK:
                    print(f"      early stop at epoch {epoch+1}  "
                          f"best_val={best_val:.4f}")
                break

        if QUICK_CHECK and (epoch + 1) % 5 == 0:
            print(f"      epoch {epoch+1}/{MAX_EPOCHS}  "
                  f"val_loss={val_loss:.4f}  "
                  f"no_improve={no_improve}/{EARLY_STOP}")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def predict(model, X_imgs, windows, device):
    """
    Returns Softmax probability of class 1 (UP).
    Matches test.ipynb: predict_logit = Softmax(y_pred)[:,1]
    """
    model.eval()
    Xt   = {w: torch.tensor(X_imgs[w].astype(np.float32)/255.0)
            for w in windows}
    N    = len(next(iter(Xt.values())))
    prob = []
    with torch.no_grad():
        for start in range(0, N, BATCH_VAL):
            xb = {w: Xt[w][start:start+BATCH_VAL].to(device)
                  for w in windows}
            p  = torch.softmax(model(xb), dim=1)[:, 1]
            prob.append(p.cpu().numpy())
    return np.hstack(prob)


# ===========================================================================
# STEP 1 — Data
# ===========================================================================
print("\n[1/4] Loading data...")

data_raw       = econ_df_1[FEATURES].copy()
data_raw.index = pd.to_datetime(data_raw.index)
data_raw       = (data_raw
                  .reindex(pd.date_range(data_raw.index.min(),
                                         data_raw.index.max(), freq='D'))
                  .ffill().dropna())

global_wti = data_raw['WTI'].values.copy()
N_data     = len(data_raw)
SPLIT      = pd.Timestamp('2020-01-01')

print(f"  {data_raw.index.min().date()} → {data_raw.index.max().date()}"
      f"  ({N_data:,} days)")


# ===========================================================================
# STEP 2 — Labels
# ===========================================================================
print("\n[2/4] Computing labels...")

ret5   = (pd.Series(global_wti).shift(-HORIZON) -
          pd.Series(global_wti)) / pd.Series(global_wti)
raw_lbl = np.where(ret5.values >  THRESHOLD,  1,
          np.where(ret5.values < -THRESHOLD,  0, -1))

max_w  = max(WINDOWS)
s_idx, s_lbl, s_dates = [], [], []
for i in range(max_w, N_data - HORIZON):
    if raw_lbl[i] == -1:
        continue
    s_idx.append(i); s_lbl.append(int(raw_lbl[i]))
    s_dates.append(data_raw.index[i])

s_idx   = np.array(s_idx)
s_lbl   = np.array(s_lbl)
s_dates = pd.DatetimeIndex(s_dates)
N_S     = len(s_lbl)
bah_acc = (s_lbl == 1).mean() * 100

print(f"  Samples: {N_S:,}  "
      f"Up: {(s_lbl==1).sum():,}  Down: {(s_lbl==0).sum():,}")
print(f"  Buy & hold baseline: {bah_acc:.2f}%")


# ===========================================================================
# STEP 3 — Walk-forward backtest
# ===========================================================================
print("\n[3/4] Walk-forward backtest...")

init_tr = int(N_S * INIT_TRAIN)
n_folds = (N_S - init_tr + STEP_SIZE - 1) // STEP_SIZE
est_min = n_folds * (2 if QUICK_CHECK else 4)  # quick=2min/fold, full=4min/fold

print(f"  Init train: {init_tr:,} samples  Step: {STEP_SIZE}")
print(f"  Folds: {n_folds}  Estimated time: ~{est_min} min")
if QUICK_CHECK:
    print(f"  QUICK CHECK — testing last {n_folds} folds only")
    print(f"  If results look good → set QUICK_CHECK=False for full run")

all_true, all_pred, all_mask = [], [], []
all_dates_out, all_returns   = [], []
fold_info                    = []
next_trade                   = 0
cursor                       = init_tr

with tqdm(total=n_folds, desc="  Folds") as pbar:
    while cursor < N_S:
        te_end = min(cursor + STEP_SIZE, N_S)
        idx_tr = s_idx[:cursor];       y_tr = s_lbl[:cursor]
        idx_te = s_idx[cursor:te_end]; y_te = s_lbl[cursor:te_end]

        if len(y_tr) < MIN_TRAIN:
            cursor += STEP_SIZE; pbar.update(1); continue

        # ── Scale WTI on training rows only ──────────────────────────────────
        tr_rows = np.unique([gi - w for gi in idx_tr for w in WINDOWS])
        tr_rows = tr_rows[tr_rows >= 0]
        sc      = MinMaxScaler()
        sc.fit(global_wti[tr_rows].reshape(-1, 1))
        wti_sc  = sc.transform(global_wti.reshape(-1, 1)).ravel()

        # ── Build images ──────────────────────────────────────────────────────
        X_tr_im = build_images(wti_sc, idx_tr, WINDOWS, IMAGE_SIZE)
        X_te_im = build_images(wti_sc, idx_te, WINDOWS, IMAGE_SIZE)

        # ── 70/30 sequential split (original) ────────────────────────────────
        split   = int(len(y_tr) * TRAIN_RATIO)
        X_tr2   = {w: X_tr_im[w][:split] for w in WINDOWS}
        X_val   = {w: X_tr_im[w][split:]  for w in WINDOWS}
        y_tr2   = y_tr[:split]
        y_val   = y_tr[split:]

        try:
            # ── Train end-to-end multi-scale CNN ─────────────────────────────
            model = train_model(X_tr2, y_tr2, X_val, y_val,
                                WINDOWS, IMAGE_SIZE, DEVICE)

            # ── Predict: Softmax[:,1] then threshold 0.58 (test.ipynb) ───────
            prob       = predict(model, X_te_im, WINDOWS, DEVICE)
            preds      = (prob >= 0.5).astype(int)
            final_mask = prob > CONF_THRESHOLD    # original: predict_logit > 0.58

            # ── Non-overlapping trades ────────────────────────────────────────
            no_m   = np.zeros(len(y_te), dtype=bool)
            last_g = next_trade
            for i in range(len(y_te)):
                gi = int(idx_te[i])
                if final_mask[i] and gi >= last_g:
                    no_m[i] = True
                    last_g  = gi + HORIZON
            next_trade = last_g
            final_mask = final_mask & no_m

            # ── Returns ───────────────────────────────────────────────────────
            fold_ret = np.zeros(len(y_te))
            for i in range(len(y_te)):
                gi  = int(idx_te[i])
                p0  = global_wti[gi]
                p1  = global_wti[gi+HORIZON] if gi+HORIZON < N_data else p0
                ret = (p1 - p0) / (p0 + 1e-8)
                fold_ret[i] = ret if preds[i] == 1 else -ret

        except Exception as e:
            print(f"\n  Fold {len(fold_info)} error: {e}")
            preds      = np.full(len(y_te), -1)
            final_mask = np.zeros(len(y_te), dtype=bool)
            fold_ret   = np.zeros(len(y_te))

        fd = s_dates[cursor] if cursor < len(s_dates) else s_dates[-1]
        fa = (accuracy_score(y_te[final_mask], preds[final_mask])
              if final_mask.sum() > 0 else 0.0)
        fold_info.append({
            'fold'  : len(fold_info),
            'acc'   : fa,
            'n_pred': int(final_mask.sum()),
            'period': 'post' if fd >= SPLIT else 'pre',
            'date'  : fd,
        })

        all_true.extend(y_te.tolist())
        all_pred.extend(preds.tolist())
        all_mask.extend(final_mask.tolist())
        all_dates_out.extend(s_dates[cursor:te_end].tolist())
        all_returns.extend(fold_ret.tolist())
        cursor += STEP_SIZE; pbar.update(1)

all_true    = np.array(all_true);    all_pred  = np.array(all_pred)
all_mask    = np.array(all_mask);    all_dates = pd.DatetimeIndex(all_dates_out)
all_returns = np.array(all_returns); fold_df   = pd.DataFrame(fold_info)


# ===========================================================================
# STEP 4 — Results
# ===========================================================================
print("\n" + "="*70)
mode_str = "QUICK CHECK (last ~5 folds)" if QUICK_CHECK else "FULL BACKTEST"
print(f"  RESULTS  |  Multi-Scale CNN  |  WTI  |  {mode_str}")
print("="*70)

vm = all_mask

if vm.sum() == 0:
    print("\n  No trades passed the confidence threshold.")
    print("  Try reducing CONF_THRESHOLD from 0.58 to 0.50")
else:
    acc = accuracy_score(all_true[vm], all_pred[vm])
    f1  = f1_score(all_true[vm], all_pred[vm], average='macro', zero_division=0)
    dr  = (all_pred[vm&(all_true==0)]==0).mean() if (vm&(all_true==0)).any() else 0
    ur  = (all_pred[vm&(all_true==1)]==1).mean() if (vm&(all_true==1)).any() else 0

    print(f"\n  Predicted : {vm.sum():,} / {len(all_true):,} "
          f"({vm.mean()*100:.1f}%)")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"  Down Rec  : {dr:.3f}    Up Rec : {ur:.3f}")
    print(f"\n{classification_report(all_true[vm], all_pred[vm], target_names=['Down','Up'], digits=4)}")

    spm = np.array([d >= SPLIT for d in all_dates])
    print("  ── Period Breakdown ────────────────────────────────────────────")
    for lbl, pm in [("Pre-2020", ~spm&vm), ("Post-2020", spm&vm)]:
        if pm.sum() > 5:
            a = accuracy_score(all_true[pm], all_pred[pm])*100
            f = f1_score(all_true[pm], all_pred[pm], average='macro', zero_division=0)
            print(f"  {lbl}: Acc={a:.2f}%  F1={f:.4f}  Trades={pm.sum()}")

    tr  = all_returns[vm] - 2*TRADE_FEE
    sh  = (tr.mean()/(tr.std()+1e-8)) * np.sqrt(252/HORIZON)
    tot = tr.sum()*100
    mdd = (np.maximum.accumulate(np.cumsum(tr)) - np.cumsum(tr)).max()*100
    print(f"\n  Total return : {tot:+.2f}%")
    print(f"  Sharpe ratio : {sh:.3f}")
    print(f"  Max drawdown : {mdd:.2f}%")
    print(f"\n  Baselines — Random: 50.00%  |  Buy&hold: {bah_acc:.2f}%")

    # ── Per-fold summary ──────────────────────────────────────────────────
    print(f"\n  ── Fold accuracy ────────────────────────────────────────────")
    print(f"  Mean : {fold_df['acc'].mean()*100:.2f}%  "
          f"Std : {fold_df['acc'].std()*100:.2f}%  "
          f"Min : {fold_df['acc'].min()*100:.2f}%  "
          f"Max : {fold_df['acc'].max()*100:.2f}%")
    for _, row in fold_df.iterrows():
        bar  = '█' * int(row['acc'] * 20)
        flag = ' ← POST-2020' if row['period'] == 'post' else ''
        print(f"  Fold {int(row['fold']):2d} "
              f"[{str(row['date'])[:10]}]  "
              f"{row['acc']*100:5.1f}%  {bar}{flag}")

    # ── Decision helper ───────────────────────────────────────────────────
    if QUICK_CHECK:
        mean_acc = fold_df['acc'].mean()
        print(f"\n  {'='*50}")
        if mean_acc >= 0.54:
            print(f"  ✅ Mean accuracy {mean_acc*100:.1f}% ≥ 54%")
            print(f"  Signal looks promising.")
            print(f"  → Set QUICK_CHECK = False to run full 72-fold backtest")
        elif mean_acc >= 0.51:
            print(f"  ⚠️  Mean accuracy {mean_acc*100:.1f}% — marginal signal")
            print(f"  → Consider transfer learning or smaller model before full run")
        else:
            print(f"  ❌ Mean accuracy {mean_acc*100:.1f}% — no signal detected")
            print(f"  → CNN is overfitting on small data")
            print(f"  → Recommendation: use transfer learning from pretrained weights")
            print(f"  → Pretrained weights available at:")
            print(f"     Stock_CNN-main/pt/baseline_epoch_5_train_0.704243_val_0.695599.pt")
        print(f"  {'='*50}")

    # ── Chart ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor='white')

    # Fold accuracy bars
    ax1 = axes[0]; ax1.set_facecolor('white')
    fa   = fold_df['acc'].values * 100
    cols = ['#A371F7' if r['period']=='post' else '#1A5276'
            for _, r in fold_df.iterrows()]
    ax1.bar(range(len(fa)), fa, color=cols, alpha=0.85, width=0.7)
    ax1.axhline(50,       color='grey', ls='--', lw=0.9, label='50% random')
    ax1.axhline(bah_acc,  color='orange', ls=':', lw=0.9,
                label=f'Buy&hold {bah_acc:.1f}%')
    ax1.axhline(acc*100,  color='red',  ls=':',  lw=1.2,
                label=f'Overall {acc*100:.1f}%')
    ax1.set_xlabel('Fold'); ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'Fold Accuracy — {mode_str}\n'
                  f'Blue=Pre-2020  Purple=Post-2020', fontweight='bold')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.4, axis='y')

    # Cumulative return
    ax2 = axes[1]; ax2.set_facecolor('white')
    cum = np.cumsum(all_returns[vm] - 2*TRADE_FEE) * 100
    ax2.plot(cum, color='#1A5276', linewidth=1.2)
    ax2.fill_between(range(len(cum)), 0, cum,
                     where=(cum >= 0), color='#1E8449', alpha=0.2)
    ax2.fill_between(range(len(cum)), 0, cum,
                     where=(cum < 0),  color='#C0392B', alpha=0.2)
    ax2.axhline(0, color='grey', lw=0.8, ls='--')
    ax2.set_xlabel('Trade number'); ax2.set_ylabel('Cumulative return (%)')
    ax2.set_title(f'Cumulative Return  |  Sharpe={sh:.2f}',
                  fontweight='bold')
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    fname = 'wti_cnn_quick.png' if QUICK_CHECK else 'wti_cnn_full.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"\n  Chart saved → {fname}")

print("\n  Done.")
if QUICK_CHECK:
    print("  Next step: check the decision above, then set QUICK_CHECK = False")
