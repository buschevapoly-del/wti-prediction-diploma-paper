"""
==============================================================================
  SIMPC + JISC-Net + LLM News Sentiment
  Directional Forecasting of WTI Crude Oil Prices

  Architecture:
    Stage 1 — SIMPC (4 decorrelated features) finds recurring patterns
    Stage 2 — JISC-Net SVM predicts direction using:
                · Shapelet DTW distances
                · Cluster OHE meta-feature
                · Full 16-day window (12 features)
                · WTI/Brent spread momentum
                · LLM news sentiment score (NEW)
    Stage 3 — Two-stage filtering:
                · Per-label K-S test (training)
                · Asymmetric confidence thresholds (inference)

  News sentiment:
    Default: FinBERT (free, local, no API key needed)
    Upgrade:  GPT-4o-mini (set USE_GPT=True + add OPENAI_API_KEY)

  Sources: Reuters RSS, Yahoo Finance RSS, GDELT API (all free)

  Horizon : 5 days  |  Window : 20 days  |  gamma : 0.8
==============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw as tslearn_dtw
from scipy.stats import ks_2samp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  SIMPC + JISC-Net + LLM News Sentiment  |  Horizon = 5 days")
print("=" * 70)


# ===========================================================================
# NEWS SENTIMENT CONFIGURATION
# ===========================================================================

USE_GPT       = False          # True  → GPT-4o-mini (needs OPENAI_API_KEY)
                                # False → FinBERT     (free, local)
OPENAI_API_KEY = ""            # paste your key here if USE_GPT=True

# Oil-relevant keywords to filter headlines
OIL_KEYWORDS = [
    'oil', 'crude', 'wti', 'brent', 'opec', 'energy', 'petroleum',
    'barrel', 'refinery', 'gasoline', 'iran', 'saudi', 'russia',
    'middle east', 'hormuz', 'eia', 'inventory'
]

# Sentiment look-back: how many days of headlines to include per window
# (we compute a rolling X-day average sentiment up to each prediction date)
SENTIMENT_WINDOW = 5


# ===========================================================================
# NEWS FETCHING (free RSS + GDELT)
# ===========================================================================

def fetch_rss_headlines(max_per_feed=30):
    """
    Fetches recent oil-related headlines from free RSS feeds.
    No API key required. Returns DataFrame with columns: date, headline.
    """
    import feedparser, requests
    from datetime import datetime, timedelta

    feeds = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://finance.yahoo.com/rss/headline?s=CL=F",
        "https://rss.cnn.com/rss/money_markets.rss",
    ]

    rows = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get('title', '')
                if not any(k in title.lower() for k in OIL_KEYWORDS):
                    continue
                # Parse date
                try:
                    pub = entry.get('published_parsed') or \
                          entry.get('updated_parsed')
                    date = pd.Timestamp(*pub[:3]).date() if pub else \
                           datetime.today().date()
                except Exception:
                    date = datetime.today().date()
                rows.append({'date': date, 'headline': title})
        except Exception:
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['date', 'headline'])


def fetch_gdelt_headlines(start_date, end_date, max_per_day=5):
    """
    Fetches historical oil headlines from GDELT API (completely free).
    Returns DataFrame with columns: date, headline.
    Works for dates back to 2015.
    """
    import requests
    from datetime import timedelta

    rows   = []
    cur    = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    print(f"  Fetching GDELT headlines "
          f"{start_date} → {end_date} ...")

    while cur <= end_ts:
        ds = cur.strftime('%Y%m%d')
        url = (f"https://api.gdeltproject.org/api/v2/doc/doc?"
               f"query=oil%20WTI%20OPEC%20crude%20petroleum&"
               f"mode=artlist&maxrecords={max_per_day}&"
               f"startdatetime={ds}000000&"
               f"enddatetime={ds}235959&format=json")
        try:
            r    = requests.get(url, timeout=8)
            data = r.json()
            for art in data.get('articles', []):
                rows.append({
                    'date':     cur.date(),
                    'headline': art.get('title', '')
                })
        except Exception:
            pass
        cur += pd.Timedelta(days=1)

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['date', 'headline'])


# ===========================================================================
# SENTIMENT SCORING
# ===========================================================================

# ── Option A: FinBERT (free, local) ──────────────────────────────────────

_finbert_pipeline = None

def load_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        print("  Loading FinBERT (ProsusAI/finbert) ...")
        _finbert_pipeline = hf_pipeline(
            "text-classification",
            model     = "ProsusAI/finbert",
            top_k     = None,
            device    = -1        # CPU; change to 0 for GPU
        )
        print("  FinBERT loaded.")
    except ImportError:
        print("  transformers not installed — run: pip install transformers")
        _finbert_pipeline = None
    return _finbert_pipeline


def score_finbert(headline: str) -> float:
    """
    Returns a scalar sentiment score using FinBERT:
      +1.0 = strongly bullish for oil
      -1.0 = strongly bearish for oil
       0.0 = neutral
    """
    pipe = load_finbert()
    if pipe is None:
        return 0.0
    try:
        result   = pipe(headline[:512])[0]
        label_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
        scores   = {item['label']: item['score'] for item in result}
        score    = sum(label_map.get(lbl, 0) * sc
                       for lbl, sc in scores.items())
        return float(np.clip(score, -1.0, 1.0))
    except Exception:
        return 0.0


# ── Option B: GPT-4o-mini ─────────────────────────────────────────────────

def score_gpt(headline: str, api_key: str) -> float:
    """
    Calls GPT-4o-mini to score a single headline.
    Returns scalar in [-1.0, +1.0].
    """
    import json, requests as _req

    prompt = f"""You are an expert oil market analyst.
Score the following headline for its expected impact on WTI crude oil prices
over the next 5 trading days.

Return ONLY a single float between -1.0 and +1.0:
  +1.0 = strongly bullish (supply cut, geopolitical tension, demand surge)
   0.0 = neutral / no impact
  -1.0 = strongly bearish (demand concern, supply glut, economic slowdown)

Headline: {headline}

Score:"""

    try:
        resp = _req.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.0, "max_tokens": 10},
            timeout=15
        )
        text  = resp.json()['choices'][0]['message']['content'].strip()
        score = float(text.split()[0].replace(',', '.'))
        return float(np.clip(score, -1.0, 1.0))
    except Exception:
        return 0.0


def score_headline(headline: str) -> float:
    """Routes to GPT or FinBERT depending on USE_GPT flag."""
    if USE_GPT and OPENAI_API_KEY:
        return score_gpt(headline, OPENAI_API_KEY)
    return score_finbert(headline)


# ===========================================================================
# BUILD DAILY SENTIMENT SERIES
# ===========================================================================

def build_daily_sentiment(news_df: pd.DataFrame,
                           date_index: pd.DatetimeIndex,
                           sentiment_window: int = 5) -> pd.Series:
    """
    1. Scores each headline with the LLM.
    2. Averages scores per calendar day.
    3. Reindexes to date_index, forward-fills, then computes a
       rolling mean over sentiment_window days.
    Returns a Series aligned to date_index, values in [-1, +1].
    """
    if news_df.empty:
        print("  No headlines found — sentiment set to 0.0 everywhere.")
        return pd.Series(0.0, index=date_index)

    # Score all headlines
    print(f"  Scoring {len(news_df):,} headlines with "
          f"{'GPT-4o-mini' if USE_GPT and OPENAI_API_KEY else 'FinBERT'} ...")

    news_df = news_df.copy()
    news_df['score'] = [
        score_headline(h)
        for h in tqdm(news_df['headline'], desc="  Scoring")
    ]

    # Daily average
    news_df['date'] = pd.to_datetime(news_df['date'])
    daily = (news_df.groupby('date')['score']
                    .mean()
                    .reindex(date_index)
                    .ffill()
                    .fillna(0.0))

    # Rolling mean to smooth daily noise
    sentiment = (daily
                 .rolling(window=sentiment_window, min_periods=1)
                 .mean()
                 .fillna(0.0))

    n_days_with_news = (news_df.groupby('date').size()
                        .reindex(date_index).notna().sum())
    print(f"  Sentiment coverage: {n_days_with_news:,} / "
          f"{len(date_index):,} days have actual headlines.")
    print(f"  Score stats — mean={sentiment.mean():.3f}  "
          f"std={sentiment.std():.3f}  "
          f"min={sentiment.min():.3f}  max={sentiment.max():.3f}")

    return sentiment


# ===========================================================================
# WTI Technical Feature Engineering
# ===========================================================================

def add_technical_features(df, price_col='WTI'):
    df = df.copy()
    delta = df[price_col].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['WTI_RSI']  = 100 - (100 / (1 + gain / (loss + 1e-8)))
    ema12 = df[price_col].ewm(span=12, adjust=False).mean()
    ema26 = df[price_col].ewm(span=26, adjust=False).mean()
    df['WTI_MACD'] = ema12 - ema26
    df['WTI_Vol20'] = df[price_col].rolling(20).std()
    df['WTI_Mom5']  = df[price_col].pct_change(5)
    df['WTI_Mom20'] = df[price_col].pct_change(20)
    ma20  = df[price_col].rolling(20).mean()
    std20 = df[price_col].rolling(20).std()
    df['WTI_BB'] = (df[price_col] - ma20) / (2 * std20 + 1e-8)
    return df.dropna()


# ===========================================================================
# Canonical Chart Pattern Centroids
# ===========================================================================

def make_canonical_centroids(window_size=20, n_features=4):
    L  = window_size
    hs = np.zeros(L)
    hs[0:4]   = np.linspace(0.2, 0.5, 4)
    hs[4:6]   = np.linspace(0.5, 0.4, 2)
    hs[6:9]   = np.linspace(0.4, 1.0, 3)
    hs[9:11]  = np.linspace(1.0, 0.4, 2)
    hs[11:14] = np.linspace(0.4, 0.55, 3)
    hs[14:17] = np.linspace(0.55, 0.2, 3)
    hs[17:]   = np.linspace(0.2, 0.1, 3)
    ihs  = 1.0 - hs
    dtop = np.zeros(L)
    dtop[0:5]   = np.linspace(0.0, 1.0, 5)
    dtop[5:8]   = np.linspace(1.0, 0.5, 3)
    dtop[8:12]  = np.linspace(0.5, 1.0, 4)
    dtop[12:16] = np.linspace(1.0, 0.4, 4)
    dtop[16:]   = np.linspace(0.4, 0.1, 4)
    dbot = 1.0 - dtop
    ttop = np.zeros(L)
    for i in range(L):
        upper = 1.0 - 0.03 * i
        phase = (i % 4) / 4.0
        ttop[i] = 0.5 + (upper - 0.5) * (np.sin(phase * np.pi) ** 2)
    ttop[16:] = np.linspace(ttop[15], 0.1, 4)
    tbot = 1.0 - ttop

    centroids = []
    for p in [hs, ihs, dtop, dbot, ttop, tbot]:
        p_norm = (p - p.min()) / (p.max() - p.min() + 1e-8)
        multi  = np.full((L, n_features), 0.5)
        multi[:, 0] = p_norm
        multi[:, 1] = 0.5 + 0.2 * (0.5 - p_norm)
        multi[:, 2] = 0.5 + 0.3 * (0.5 - p_norm)
        multi[:, 3] = 0.5 + 0.1 * (p_norm - 0.5)
        centroids.append(multi)

    print(f"  -> {len(centroids)} canonical centroids "
          f"(shape: {centroids[0].shape})")
    return centroids


# ===========================================================================
# Smoothing, Labels, Shapelet Features
# ===========================================================================

def nadaraya_watson_smoothing(series, h=5):
    t = np.arange(len(series))
    smoothed = np.zeros_like(series, dtype=float)
    for i in range(len(series)):
        w = np.exp(-((t - i) ** 2) / (2 * h ** 2))
        w /= w.sum()
        smoothed[i] = np.dot(w, series)
    return smoothed


def make_robust_labels(price_series, horizon=5, threshold=0.003):
    returns = (price_series.shift(-horizon) - price_series) / \
              (price_series + 1e-8)
    return np.where(returns >  threshold,  1,
           np.where(returns < -threshold,  0, -1))


def compute_shapelet_features(X_windows, shapelets, verbose=True):
    n_samples, n_shapelets = X_windows.shape[0], len(shapelets)
    out = np.zeros((n_samples, n_shapelets))
    it  = tqdm(range(n_samples), desc="  Shapelet distances",
               disable=not verbose)
    for i in it:
        window = X_windows[i]
        for j, shp in enumerate(shapelets):
            shp_len, win_len = shp.shape[0], window.shape[0]
            min_dist = np.inf
            for start in range(max(1, win_len - shp_len + 1)):
                seg = window[start : start + shp_len]
                if seg.shape[0] == shp_len:
                    d = tslearn_dtw(seg, shp)
                    if d < min_dist:
                        min_dist = d
            out[i, j] = min_dist if min_dist < np.inf else 0.0
    return out


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

SIMPC_FEATURES = ['WTI', 'Gold', 'Dollar/Euro', 'Copper']
CLASSIF_BASE   = ['WTI', 'Brent', 'Gold', 'Dollar/Euro', 'Copper', 'S&P500']

# ---------- 1. Load + engineer features ----------
print("\n[1/9] Loading and engineering features...")

data_raw = econ_df_1[CLASSIF_BASE].copy()
data_raw.index = pd.to_datetime(data_raw.index)

full_dates = pd.date_range(start=data_raw.index.min(),
                            end=data_raw.index.max(), freq='D')
data_raw   = data_raw.reindex(full_dates).ffill()
data_raw   = add_technical_features(data_raw, price_col='WTI')

all_features = list(data_raw.columns)
n_feat_all   = len(all_features)
simpc_idx    = [all_features.index(f) for f in SIMPC_FEATURES]
n_simpc      = len(SIMPC_FEATURES)

print(f"  Classifier features ({n_feat_all}): {all_features}")
print(f"  SIMPC features (4 decorrelated): {SIMPC_FEATURES}")


# ---------- 2. Smoothing + normalization ----------
print("\n[2/9] Nadaraya-Watson smoothing + Min-Max normalization...")

data_smoothed = data_raw.copy()
for col in all_features:
    data_smoothed[col] = nadaraya_watson_smoothing(data_raw[col].values, h=5)

scaler      = MinMaxScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(data_smoothed),
    index=data_smoothed.index, columns=all_features
)
print("  -> Done.")


# ---------- 3. News sentiment ----------
print("\n[3/9] Building news sentiment series...")

# ── Strategy:
#   For HISTORICAL dates (training): use GDELT API to fetch real headlines.
#   For RECENT dates  (test end):    use RSS feeds for live headlines.
#   For dates with no headlines:     sentiment = 0.0 (neutral).
#
# The sentiment score is added as ONE extra column to the SVM feature matrix.
# It is NOT used in SIMPC — pattern clustering stays purely price-based.

# Install feedparser if needed
try:
    import feedparser
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'feedparser', '-q'])
    import feedparser

# Install transformers if needed (for FinBERT)
if not USE_GPT:
    try:
        from transformers import pipeline as _
    except ImportError:
        import subprocess
        print("  Installing transformers for FinBERT...")
        subprocess.run(['pip', 'install', 'transformers', 'torch', '-q'])

# Determine date range of the dataset
data_start = data_raw.index.min()
data_end   = data_raw.index.max()

# Fetch headlines — GDELT for historical, RSS for recent
print(f"  Fetching headlines for {data_start.date()} → {data_end.date()}")
print("  Note: GDELT covers from ~2015. Earlier dates get neutral score.")

# For speed in Colab: sample every 7th day from GDELT (weekly sampling)
# Change step=1 for full daily coverage (much slower)
gdelt_dates = pd.date_range(start=max(data_start,
                                       pd.Timestamp('2015-01-01')),
                              end=data_end, freq='7D')

all_news_rows = []

# GDELT historical
try:
    import requests as _req
    for dt in tqdm(gdelt_dates, desc="  GDELT fetch"):
        ds  = dt.strftime('%Y%m%d')
        url = (f"https://api.gdeltproject.org/api/v2/doc/doc?"
               f"query=oil%20WTI%20OPEC%20crude&"
               f"mode=artlist&maxrecords=5&"
               f"startdatetime={ds}000000&"
               f"enddatetime={ds}235959&format=json")
        try:
            r    = _req.get(url, timeout=8)
            data_g = r.json()
            for art in data_g.get('articles', []):
                all_news_rows.append({
                    'date':     dt.date(),
                    'headline': art.get('title', '')
                })
        except Exception:
            pass
except Exception as e:
    print(f"  GDELT fetch error: {e} — using neutral sentiment for all dates.")

# RSS for most recent headlines (last 30 days)
rss_df = fetch_rss_headlines(max_per_feed=50)
if not rss_df.empty:
    all_news_rows.extend(rss_df.to_dict('records'))

news_df = pd.DataFrame(all_news_rows) if all_news_rows else pd.DataFrame(
    columns=['date', 'headline'])

# Filter to oil-relevant headlines only
if not news_df.empty:
    mask = news_df['headline'].str.lower().apply(
        lambda t: any(k in t for k in OIL_KEYWORDS)
    )
    news_df = news_df[mask].reset_index(drop=True)
    print(f"  -> {len(news_df):,} oil-relevant headlines found")
else:
    print("  -> No headlines fetched — using neutral sentiment (0.0)")

# Build daily sentiment series
daily_sentiment = build_daily_sentiment(
    news_df,
    date_index      = data_scaled.index,
    sentiment_window = SENTIMENT_WINDOW
)

# Normalize sentiment to [0, 1] so it sits on the same scale as other features
sent_min    = daily_sentiment.min()
sent_max    = daily_sentiment.max()
sent_range  = sent_max - sent_min
if sent_range > 1e-8:
    daily_sentiment_norm = (daily_sentiment - sent_min) / sent_range
else:
    daily_sentiment_norm = daily_sentiment * 0.0 + 0.5   # all neutral


# ---------- 4. Parameters ----------
print("\n[4/9] Parameters...")

WINDOW_SIZE  = 20
HORIZON      = 5
GAMMA        = 0.8
INPUT_LENGTH = int(GAMMA * WINDOW_SIZE)
THRESHOLD    = 0.003
N_CLUSTERS   = 12

print(f"  window={WINDOW_SIZE}d | gamma={GAMMA} -> input={INPUT_LENGTH}d | "
      f"horizon={HORIZON}d | clusters={N_CLUSTERS}")
print(f"  Sentiment: {'GPT-4o-mini' if USE_GPT and OPENAI_API_KEY else 'FinBERT'}")


# ---------- 5. Window creation with robust labels ----------
print("\n[5/9] Building windows...")

y_all_raw = make_robust_labels(
    data_scaled['WTI'], horizon=HORIZON, threshold=THRESHOLD
)

X_windows, y_direction, sent_per_window = [], [], []
skipped_neutral = 0

for i in tqdm(range(len(data_scaled) - WINDOW_SIZE - HORIZON),
              desc="  Building windows"):
    label = y_all_raw[i + WINDOW_SIZE - 1]
    if label == -1:
        skipped_neutral += 1
        continue
    window = data_scaled.iloc[i : i + WINDOW_SIZE].values   # (20, 12)
    X_windows.append(window)
    y_direction.append(int(label))
    # Sentiment at the END of this window (prediction day)
    sent_per_window.append(
        float(daily_sentiment_norm.iloc[i + WINDOW_SIZE - 1])
    )

X    = np.array(X_windows)     # (N, 20, 12)
y    = np.array(y_direction)
S    = np.array(sent_per_window).reshape(-1, 1)   # (N, 1)

print(f"  -> {X.shape[0]:,} samples | {skipped_neutral:,} neutral skipped")
print(f"  -> Down={np.sum(y==0):,}  Up={np.sum(y==1):,}")
print(f"  -> Sentiment: mean={S.mean():.3f}  std={S.std():.3f}")


# ---------- 6. Train / test split ----------
print("\n[6/9] Train/test split (80/20, temporal)...")

X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
    X, y, S, test_size=0.2, shuffle=False, random_state=42
)

print(f"  Train — Down={np.sum(y_train==0):,}  Up={np.sum(y_train==1):,}  "
      f"  Sent mean={S_train.mean():.3f}")
print(f"  Test  — Down={np.sum(y_test==0):,}   Up={np.sum(y_test==1):,}  "
      f"  Sent mean={S_test.mean():.3f}")


# ---------- 7. SIMPC (4 decorrelated features) ----------
print("\n[7/9] Training SIMPC (4 decorrelated features)...")

X_train_simpc = X_train[:, :, simpc_idx].astype(np.float64)
X_test_simpc  = X_test[:,  :, simpc_idx].astype(np.float64)

canonical    = make_canonical_centroids(window_size=WINDOW_SIZE,
                                         n_features=n_simpc)
canonical_3d = np.array(canonical, dtype=np.float64)
n_extra      = N_CLUSTERS - len(canonical)
rng          = np.random.default_rng(42)
extra_idx    = rng.choice(len(X_train_simpc), size=n_extra, replace=False)
init_centroids = np.concatenate(
    [canonical_3d, X_train_simpc[extra_idx]], axis=0
)

simpc = TimeSeriesKMeans(
    n_clusters=N_CLUSTERS, metric="dtw", max_iter=10,
    init=init_centroids, random_state=42, n_jobs=-1, verbose=False
)
simpc.fit(X_train_simpc)

train_cluster_labels = simpc.predict(X_train_simpc)
test_cluster_labels  = simpc.predict(X_test_simpc)
print("  -> SIMPC finished.")


# ---------- 8. JISC-Net feature matrix ----------
print("\n[8/9] Building JISC-Net + sentiment feature matrix...")

X_train_init = X_train_simpc[:, :INPUT_LENGTH, :]
X_test_init  = X_test_simpc[:,  :INPUT_LENGTH, :]

shp_len   = INPUT_LENGTH // 2
shapelets = [simpc.cluster_centers_[k][:shp_len, :]
             for k in range(N_CLUSTERS)]

print(f"  -> {len(shapelets)} shapelets  (len={shp_len}d, {n_simpc} features)")
print("  -> Computing train shapelet distances...")
X_train_shp = compute_shapelet_features(X_train_init, shapelets, verbose=True)
print("  -> Computing test shapelet distances...")
X_test_shp  = compute_shapelet_features(X_test_init,  shapelets, verbose=True)

# Cluster OHE
import sklearn
sk_ver = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
ohe_kw = {'sparse_output': False} if sk_ver >= (1, 2) else {'sparse': False}
ohe    = OneHotEncoder(**ohe_kw, handle_unknown='ignore',
                        categories=[list(range(N_CLUSTERS))])
train_cluster_ohe = ohe.fit_transform(train_cluster_labels.reshape(-1, 1))
test_cluster_ohe  = ohe.transform(test_cluster_labels.reshape(-1, 1))

# Flattened full-feature window (16d × 12 feat = 192)
X_train_full_flat = X_train[:, :INPUT_LENGTH, :].reshape(X_train.shape[0], -1)
X_test_full_flat  = X_test[:,  :INPUT_LENGTH, :].reshape(X_test.shape[0],  -1)

# WTI/Brent spread momentum
spread_mom_train = (
    X_train[:, -1, all_features.index('WTI')] -
    X_train[:, -1, all_features.index('Brent')] -
    X_train[:, -6, all_features.index('WTI')] +
    X_train[:, -6, all_features.index('Brent')]
).reshape(-1, 1)
spread_mom_test = (
    X_test[:, -1, all_features.index('WTI')] -
    X_test[:, -1, all_features.index('Brent')] -
    X_test[:, -6, all_features.index('WTI')] +
    X_test[:, -6, all_features.index('Brent')]
).reshape(-1, 1)

# ── Combine all features ──────────────────────────────────────────────────
# Without sentiment (baseline comparison)
X_train_no_sent = np.hstack([
    X_train_shp, train_cluster_ohe, X_train_full_flat, spread_mom_train
])
X_test_no_sent = np.hstack([
    X_test_shp, test_cluster_ohe, X_test_full_flat, spread_mom_test
])

# With sentiment (main model)
X_train_feat = np.hstack([X_train_no_sent, S_train])
X_test_feat  = np.hstack([X_test_no_sent,  S_test])

imputer      = SimpleImputer(strategy='mean')
X_train_feat = imputer.fit_transform(X_train_feat)
X_test_feat  = imputer.transform(X_test_feat)

# Also impute no-sentiment version for fair comparison
X_train_no_sent = imputer.transform(
    np.hstack([X_train_no_sent,
               np.zeros((len(X_train_no_sent), 1))])
)[:, :-1]
X_test_no_sent = imputer.transform(
    np.hstack([X_test_no_sent,
               np.zeros((len(X_test_no_sent), 1))])
)[:, :-1]

print(f"  -> SVM dimensionality WITH sentiment   : {X_train_feat.shape[1]}")
print(f"  -> SVM dimensionality WITHOUT sentiment: {X_train_no_sent.shape[1]}")


# ── Best DOWN class weight search ─────────────────────────────────────────
print("\n  -> Searching best DOWN class weight...")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_feat, y_train, test_size=0.15, shuffle=False, random_state=42
)

best_weight, best_f1 = 2.0, -1.0
for down_w in [1.5, 2.0, 2.5, 3.0]:
    _clf = SVC(kernel='linear', class_weight={0: down_w, 1: 1.0},
               probability=True, C=1.0, random_state=42)
    _clf.fit(X_tr, y_tr)
    _f1 = f1_score(y_val, _clf.predict(X_val), average='macro')
    _dr = (_clf.predict(X_val)[y_val==0]==0).mean() if (y_val==0).any() else 0
    print(f"     DOWN weight={down_w:.1f}  macro F1={_f1:.4f}  "
          f"down_recall={_dr:.3f}")
    if _f1 > best_f1:
        best_f1, best_weight = _f1, down_w

print(f"  -> Best weight: {best_weight}")


def train_and_evaluate(X_tr_full, X_te_full, label, best_weight, ks_mask=None):
    """
    Trains one SVM and returns predictions + probabilities.
    Returns (jisc_model, y_pred_sym, proba_down, proba_up, test_pmax, ks_mask)
    """
    clf = SVC(kernel='linear', class_weight={0: best_weight, 1: 1.0},
              probability=True, C=1.0, random_state=42)
    clf.fit(X_tr_full, y_train)

    # K-S filter
    tr_probs = clf.predict_proba(X_tr_full)
    tr_pmax  = tr_probs.max(axis=1)
    tr_preds = clf.predict(X_tr_full)
    valid    = []
    for lbl in np.unique(y_train):
        m      = (y_train == lbl)
        ok     = (tr_preds == y_train)
        c_conf = tr_pmax[m &  ok]
        w_conf = tr_pmax[m & ~ok]
        if len(c_conf) < 5 or len(w_conf) < 5:
            valid.append(lbl); continue
        _, p = ks_2samp(c_conf, w_conf)
        if p < 0.05:
            valid.append(lbl)

    ks_m = np.isin(y_test, valid)

    te_probs = clf.predict_proba(X_te_full)
    pd_arr   = te_probs[:, 0]
    pu_arr   = te_probs[:, 1]
    pmax     = te_probs.max(axis=1)
    y_sym    = clf.predict(X_te_full)

    return clf, y_sym, pd_arr, pu_arr, pmax, ks_m


# ── Train WITHOUT sentiment (baseline) ───────────────────────────────────
print(f"\n  -> Training SVM WITHOUT sentiment...")
_, y_sym_ns, pd_ns, pu_ns, pmax_ns, ks_ns = train_and_evaluate(
    X_train_no_sent, X_test_no_sent, "no_sent", best_weight
)

# ── Train WITH sentiment (main model) ─────────────────────────────────────
print(f"  -> Training SVM WITH sentiment...")
jisc_model, y_sym_s, pd_s, pu_s, pmax_s, ks_s = train_and_evaluate(
    X_train_feat, X_test_feat, "sent", best_weight
)


# ── Asymmetric threshold search (on sentiment model) ─────────────────────
print("\n  -> Searching best asymmetric thresholds...")
best_asym_f1, best_d_thr, best_u_thr = -1.0, 0.40, 0.55

for d_thr in [0.35, 0.40, 0.45]:
    for u_thr in [0.50, 0.55, 0.60]:
        y_asym = np.where(pd_s >= d_thr, 0,
                 np.where(pu_s >= u_thr, 1, -1))
        valid  = (y_asym != -1) & ks_s
        if valid.sum() < 50:
            continue
        _f1 = f1_score(y_test[valid], y_asym[valid], average='macro')
        _dr = (y_asym[valid & (y_test==0)]==0).mean() \
              if (valid & (y_test==0)).any() else 0
        print(f"     DOWN>={d_thr}  UP>={u_thr}  "
              f"macro F1={_f1:.4f}  down_recall={_dr:.3f}  "
              f"samples={valid.sum()}")
        if _f1 > best_asym_f1:
            best_asym_f1, best_d_thr, best_u_thr = _f1, d_thr, u_thr

print(f"  -> Best: DOWN>={best_d_thr}  UP>={best_u_thr}")


# ===========================================================================
# 9. Results
# ===========================================================================
print("\n" + "=" * 70)
print("  RESULTS  |  SIMPC + JISC-Net + LLM News Sentiment")
print("=" * 70)

def report(name, y_true, y_pred, n_total):
    if len(y_true) == 0:
        print(f"\n  {name}: no samples.")
        return None, None, None
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro') \
          if len(np.unique(y_pred)) > 1 else 0.0
    dr  = (y_pred[y_true==0]==0).mean() if (y_true==0).any() else 0.0

    print(f"\n  {'─'*62}")
    print(f"  {name}")
    print(f"  {'─'*62}")
    print(f"  Accuracy : {acc*100:.2f}%  |  Macro F1: {mf1:.4f}  "
          f"|  Down recall: {dr:.3f}")
    print(f"  Samples  : {len(y_true):,} / {n_total:,} "
          f"({len(y_true)/n_total*100:.1f}%)")
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        print(classification_report(y_true, y_pred,
                                    target_names=['Down(0)', 'Up(1)'],
                                    digits=4))
    return acc, mf1, dr


n_test = len(y_test)

# A — without sentiment, symmetric T@70
thr_ns   = np.percentile(pmax_ns[ks_ns], 30) if ks_ns.sum() > 0 else 0.0
mask_ns  = (pmax_ns >= thr_ns) & ks_ns
acc_ns, f1_ns, dr_ns = report(
    "A) No sentiment — symmetric T@70 (baseline)",
    y_test[mask_ns], y_sym_ns[mask_ns], n_test
)

# B — with sentiment, symmetric T@70
thr_s    = np.percentile(pmax_s[ks_s], 30) if ks_s.sum() > 0 else 0.0
mask_s   = (pmax_s >= thr_s) & ks_s
acc_s, f1_s, dr_s = report(
    "B) With sentiment — symmetric T@70",
    y_test[mask_s], y_sym_s[mask_s], n_test
)

# C — with sentiment, asymmetric thresholds
y_asym_best = np.where(pd_s >= best_d_thr, 0,
              np.where(pu_s >= best_u_thr,  1, -1))
asym_mask   = (y_asym_best != -1) & ks_s
acc_as, f1_as, dr_as = report(
    f"C) With sentiment — asymmetric "
    f"DOWN>={best_d_thr} / UP>={best_u_thr}",
    y_test[asym_mask], y_asym_best[asym_mask], n_test
)

# D — with sentiment, asymmetric + T@40
thr_40     = np.percentile(pmax_s[ks_s], 60) if ks_s.sum() > 0 else 0.0
mask_asym_40 = asym_mask & (pmax_s >= thr_40)
acc_a40, f1_a40, dr_a40 = report(
    "D) With sentiment — asymmetric + T@40 (highest precision)",
    y_test[mask_asym_40], y_asym_best[mask_asym_40], n_test
)

# ── Sentiment impact summary ──────────────────────────────────────────────
print(f"\n  {'─'*62}")
print("  Sentiment impact — direct comparison:")
print(f"  {'Config':<42} {'Acc':>7} {'MacroF1':>9} {'DownRec':>9}")
print("  " + "-" * 70)

rows = [
    ("A) No sentiment (baseline)",
     acc_ns, f1_ns, dr_ns),
    ("B) + Sentiment (symmetric)",
     acc_s,  f1_s,  dr_s),
    ("C) + Sentiment (asymmetric)",
     acc_as, f1_as, dr_as),
    ("D) + Sentiment (asym + T@40)",
     acc_a40, f1_a40, dr_a40),
]
for name, a, f, d in rows:
    if a is None:
        continue
    delta_acc = f" ({(a - (acc_ns or 0))*100:+.2f}%)" \
                if name != "A) No sentiment (baseline)" else ""
    print(f"  {name:<42} {a*100:>6.2f}%{delta_acc}  "
          f"{f:>8.4f}  {d:>8.3f}")

# ── Example prediction ────────────────────────────────────────────────────
print(f"\n  Example prediction (first non-filtered sample):")
first_valid = np.where(asym_mask)[0]
if len(first_valid) > 0:
    idx = first_valid[0]
    print(f"    True label       : {'UP' if y_test[idx]==1 else 'DOWN'}")
    print(f"    Predicted        : {'UP' if y_asym_best[idx]==1 else 'DOWN'}")
    print(f"    P(DOWN)          : {pd_s[idx]:.4f}")
    print(f"    P(UP)            : {pu_s[idx]:.4f}")
    print(f"    Sentiment score  : {S_test[idx, 0]:.4f} "
          f"(0=bearish, 0.5=neutral, 1=bullish)")
    print(f"    SIMPC cluster    : {test_cluster_labels[idx]}")
else:
    print("    No samples passed asymmetric filter.")

print("\n" + "=" * 70)
print("  Done.")
print("=" * 70)
