"""
Data Loading Pipeline — WTI Forecasting
Downloads all required data from FRED and Yahoo Finance.

Run once:  python data_loading.py
Output:    econ_df_1.csv
"""
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from datetime import datetime
import time, warnings
warnings.filterwarnings("ignore")

START = datetime(2000, 1, 1)
END   = datetime.now()

FRED_SERIES = {
    "DCOILWTICO":       "WTI",
    "DEXUSEU":          "Dollar/Euro",
    "DJIA":             "DJ",
    "NASDAQCOM":        "NASDAQ",
    "DCOILBRENTEU":     "Brent",
    "CPIENGSL":         "US_CPI_energy",
    "CPIAUCSL":         "US_CPI_total",
    "PPIACO":           "US_PPI_total",
    "FEDFUNDS":         "Fed_Funds_Effective",
    "PIEAEN01EZM661N":  "EU_PPI",
}

YF_SYMBOLS = {
    "^GSPC":   "S&P500",
    "HG=F":    "Copper",
    "GC=F":    "Gold",
    "BTC-USD": "Bitcoin",
}

def fetch_fred():
    print("Downloading FRED data...")
    df = pd.DataFrame()
    for series_id, name in FRED_SERIES.items():
        for attempt in range(3):
            try:
                s = web.get_data_fred(series_id, START, END)
                s.columns = [name]
                df = s if df.empty else df.join(s, how="outer")
                print(f"  OK  {name}"); break
            except Exception as e:
                if attempt == 2: print(f"  FAIL {name}: {e}")
                time.sleep(2)
    return df

def fetch_yf():
    print("Downloading Yahoo Finance data...")
    frames = {}
    for ticker, name in YF_SYMBOLS.items():
        try:
            s = yf.download(ticker, start=START, end=END,
                            progress=False)["Close"].rename(name)
            frames[name] = s
            print(f"  OK  {name}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
    return pd.concat(frames.values(), axis=1) if frames else pd.DataFrame()

def build():
    fred = fetch_fred()
    yf_  = fetch_yf()
    fred.index = pd.to_datetime(fred.index)
    yf_.index  = pd.to_datetime(yf_.index)
    df = fred.join(yf_, how="outer")
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df  = df.reindex(idx).ffill().bfill()
    print(f"\nShape: {df.shape}  |  NaNs: {df.isna().sum().sum()}")
    df.to_csv("econ_df_1.csv")
    print("Saved  econ_df_1.csv")
    return df

if __name__ == "__main__":
    econ_df_1 = build()
