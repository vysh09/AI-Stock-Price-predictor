"""
Data fetching and stock symbol resolution utilities.
All data comes from Yahoo Finance via the `yfinance` package - fully offline
aside from the Yahoo Finance calls themselves (no paid API keys required).
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

# A small set of common name -> ticker mappings so users can type a company
# name instead of the exact Yahoo Finance symbol. This is a convenience
# lookup only - any valid Yahoo Finance ticker can always be typed directly.
NAME_TO_TICKER = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "tesla": "TSLA",
    "google": "GOOG", "alphabet": "GOOG", "meta": "META", "facebook": "META",
    "amazon": "AMZN", "netflix": "NFLX", "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS", "infosys": "INFY.NS", "hdfc bank": "HDFCBANK.NS",
    "hdfc": "HDFCBANK.NS", "icici bank": "ICICIBANK.NS", "icici": "ICICIBANK.NS",
    "sbi": "SBIN.NS", "state bank of india": "SBIN.NS",
    "tata motors": "TATAMOTORS.NS", "wipro": "WIPRO.NS",
    "adani enterprises": "ADANIENT.NS", "adani": "ADANIENT.NS",
}

TRENDING_STOCKS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "GOOG", "AMZN",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
]

PERIOD_TO_HORIZON_DAYS = {
    "30 Days": 30,
    "3 Months": 90,
    "6 Months": 182,
    "1 Year": 365,
    "3 Years": 365 * 3,
    "5 Years": 365 * 5,
    "10 Years": 365 * 10,
}


def resolve_symbol(user_input: str) -> str:
    """Resolve a free-text company name or ticker into a Yahoo Finance symbol."""
    if not user_input:
        return ""
    text = user_input.strip()
    key = text.lower()
    if key in NAME_TO_TICKER:
        return NAME_TO_TICKER[key]
    # Already looks like a ticker (letters/digits/dot/hyphen, no spaces)
    return text.upper()


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_history(symbol: str, years: int = 10) -> pd.DataFrame:
    """Download historical OHLCV data for a symbol, cached for 30 minutes."""
    period = f"{years}y" if years <= 10 else "max"
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={"adj close": "adj_close"})
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna()


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_quote_info(symbol: str) -> dict:
    """Fetch lightweight company/quote info for headline display."""
    try:
        t = yf.Ticker(symbol)
        info = t.fast_info
        return {
            "last_price": float(info.get("last_price", np.nan)),
            "previous_close": float(info.get("previous_close", np.nan)),
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("market_cap", None),
            "day_high": info.get("day_high", None),
            "day_low": info.get("day_low", None),
            "year_high": info.get("year_high", None),
            "year_low": info.get("year_low", None),
        }
    except Exception:
        return {}


def benchmark_symbol_for(symbol: str) -> str:
    """Pick a reasonable market benchmark for beta/alpha calculations."""
    if symbol.upper().endswith(".NS") or symbol.upper().endswith(".BO"):
        return "^NSEI"
    return "^GSPC"


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_top_movers(tickers=None) -> pd.DataFrame:
    """Fetch a quick daily % change snapshot for a list of tickers (for
    Top Gainers / Top Losers / trending widgets on the landing page)."""
    tickers = tickers or TRENDING_STOCKS
    rows = []
    for tkr in tickers:
        try:
            hist = yf.download(tkr, period="5d", interval="1d", auto_adjust=True, progress=False)
            if hist is None or len(hist) < 2:
                continue
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = [c[0] for c in hist.columns]
            last, prev = hist["Close"].iloc[-1], hist["Close"].iloc[-2]
            pct = (last - prev) / prev * 100
            rows.append({"symbol": tkr, "price": round(float(last), 2), "pct_change": round(float(pct), 2)})
        except Exception:
            continue
    return pd.DataFrame(rows)
