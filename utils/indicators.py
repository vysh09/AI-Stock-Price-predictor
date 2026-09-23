"""Technical indicator calculations, computed with plain pandas/numpy so the
app has no dependency on a separate TA library."""

import pandas as pd
import numpy as np


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Given an OHLCV dataframe (columns: date, open, high, low, close, volume),
    return a copy with SMA, EMA, RSI, MACD, Bollinger Bands, VWAP, and ATR added."""
    out = df.copy()

    out["sma_20"] = out["close"].rolling(20).mean()
    out["sma_50"] = out["close"].rolling(50).mean()
    out["sma_200"] = out["close"].rolling(200).mean()

    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["close"].ewm(span=26, adjust=False).mean()

    # MACD
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # RSI (14)
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    out["rsi_14"] = out["rsi_14"].fillna(50)

    # Bollinger Bands (20, 2 std)
    mid = out["close"].rolling(20).mean()
    std = out["close"].rolling(20).std()
    out["bb_mid"] = mid
    out["bb_upper"] = mid + 2 * std
    out["bb_lower"] = mid - 2 * std

    # VWAP (cumulative, resets are not modeled - approximate rolling VWAP)
    typical_price = (out["high"] + out["low"] + out["close"]) / 3
    out["vwap"] = (typical_price * out["volume"]).cumsum() / out["volume"].cumsum()

    # ATR (14)
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()

    return out


def support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """Simple pivot-style support/resistance from recent price action."""
    recent = df.tail(window)
    high, low, close = recent["high"].max(), recent["low"].min(), df["close"].iloc[-1]
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return {"pivot": pivot, "resistance_1": r1, "resistance_2": r2,
            "support_1": s1, "support_2": s2}
