"""
Multi-model stock forecasting engine.

Each `run_<model>` function takes a prepared price dataframe and a forecast
horizon (in trading days) and returns a dict:
    {
        "dates": [future dates],
        "forecast": [predicted close prices],
        "lower": [lower confidence band],
        "upper": [upper confidence band],
        "holdout_metrics": {rmse, mae, mape, r2, directional_accuracy},
    }
All models are trained on-the-fly (no pre-trained weights are shipped) so the
app works the first time it's run, purely from data downloaded via yfinance.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from .metrics import regression_metrics

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _future_business_days(last_date, n_days):
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)


def _holdout_split(close: pd.Series, holdout_frac=0.15):
    n = len(close)
    split = max(int(n * (1 - holdout_frac)), n - 60) if n > 100 else int(n * 0.85)
    split = min(split, n - 5)
    return close.iloc[:split], close.iloc[split:]


def _residual_band(residual_std, horizon, base_value=1.0):
    """Confidence band that widens with sqrt(time), a standard random-walk
    approximation for multi-step-ahead uncertainty."""
    steps = np.arange(1, horizon + 1)
    width = residual_std * np.sqrt(steps)
    return width


def _make_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    close = df["close"]
    for lag in [1, 2, 3, 5, 10, 20]:
        feat[f"lag_{lag}"] = close.shift(lag)
    feat["return_1"] = close.pct_change(1)
    feat["return_5"] = close.pct_change(5)
    feat["sma_20"] = df.get("sma_20", close.rolling(20).mean())
    feat["sma_50"] = df.get("sma_50", close.rolling(50).mean())
    feat["rsi_14"] = df.get("rsi_14", pd.Series(50, index=df.index))
    feat["macd"] = df.get("macd", pd.Series(0, index=df.index))
    feat["volatility_10"] = close.pct_change().rolling(10).std()
    feat["target"] = close
    return feat.dropna()


# --------------------------------------------------------------------------- #
# Classical ML regressors (Linear Regression, Random Forest, XGBoost, LightGBM)
# --------------------------------------------------------------------------- #

def _run_sklearn_style(df, horizon, estimator_builder, model_name):
    feat = _make_lag_features(df)
    if len(feat) < 60:
        raise ValueError(f"Not enough data to train {model_name}")

    X = feat.drop(columns=["target"])
    y = feat["target"]

    split = int(len(feat) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = estimator_builder()
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    holdout_metrics = regression_metrics(y_test.values, test_pred)
    residual_std = float(np.std(y_test.values - test_pred)) if len(y_test) > 1 else float(y.std() * 0.02)

    # Refit on all data for the actual future forecast
    final_model = estimator_builder()
    final_model.fit(X, y)

    history = df.copy().reset_index(drop=True)
    last_row = feat.iloc[[-1]].drop(columns=["target"]).copy()
    forecasts = []
    close_history = list(df["close"].values)

    for _ in range(horizon):
        pred = float(final_model.predict(last_row)[0])
        forecasts.append(pred)
        close_history.append(pred)
        s = pd.Series(close_history)
        last_row = pd.DataFrame([{
            "lag_1": s.iloc[-2], "lag_2": s.iloc[-3], "lag_3": s.iloc[-4],
            "lag_5": s.iloc[-6] if len(s) > 6 else s.iloc[0],
            "lag_10": s.iloc[-11] if len(s) > 11 else s.iloc[0],
            "lag_20": s.iloc[-21] if len(s) > 21 else s.iloc[0],
            "return_1": (s.iloc[-1] - s.iloc[-2]) / s.iloc[-2],
            "return_5": (s.iloc[-1] - s.iloc[-6]) / s.iloc[-6] if len(s) > 6 else 0.0,
            "sma_20": s.tail(20).mean(),
            "sma_50": s.tail(50).mean(),
            "rsi_14": last_row["rsi_14"].values[0],
            "macd": last_row["macd"].values[0],
            "volatility_10": s.pct_change().tail(10).std(),
        }])

    dates = _future_business_days(df["date"].iloc[-1], horizon)
    band = _residual_band(residual_std, horizon)
    return {
        "dates": list(dates),
        "forecast": forecasts,
        "lower": [f - b for f, b in zip(forecasts, band)],
        "upper": [f + b for f, b in zip(forecasts, band)],
        "holdout_metrics": holdout_metrics,
    }


def run_linear_regression(df, horizon):
    return _run_sklearn_style(df, horizon, lambda: LinearRegression(), "Linear Regression")


def run_random_forest(df, horizon):
    return _run_sklearn_style(
        df, horizon,
        lambda: RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "Random Forest",
    )


def run_xgboost(df, horizon):
    from xgboost import XGBRegressor
    return _run_sklearn_style(
        df, horizon,
        lambda: XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42),
        "XGBoost",
    )


def run_lightgbm(df, horizon):
    from lightgbm import LGBMRegressor
    return _run_sklearn_style(
        df, horizon,
        lambda: LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        "LightGBM",
    )


# --------------------------------------------------------------------------- #
# ARIMA
# --------------------------------------------------------------------------- #

def run_arima(df, horizon):
    from statsmodels.tsa.arima.model import ARIMA

    close = df["close"].reset_index(drop=True)
    train, test = _holdout_split(close)

    model = ARIMA(train, order=(5, 1, 0)).fit()
    test_pred = model.forecast(steps=len(test))
    holdout_metrics = regression_metrics(test.values, test_pred.values)

    final_model = ARIMA(close, order=(5, 1, 0)).fit()
    fc = final_model.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.2)

    dates = _future_business_days(df["date"].iloc[-1], horizon)
    return {
        "dates": list(dates),
        "forecast": list(mean.values),
        "lower": list(ci.iloc[:, 0].values),
        "upper": list(ci.iloc[:, 1].values),
        "holdout_metrics": holdout_metrics,
    }


# --------------------------------------------------------------------------- #
# Prophet
# --------------------------------------------------------------------------- #

def run_prophet(df, horizon):
    from prophet import Prophet

    hist = df[["date", "close"]].rename(columns={"date": "ds", "close": "y"})
    train, test = hist.iloc[:int(len(hist) * 0.85)], hist.iloc[int(len(hist) * 0.85):]

    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True,
                interval_width=0.8)
    m.fit(train)
    test_future = m.make_future_dataframe(periods=len(test), freq="B")
    test_fc = m.predict(test_future).tail(len(test))
    holdout_metrics = regression_metrics(test["y"].values, test_fc["yhat"].values)

    final_model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True,
                           interval_width=0.8)
    final_model.fit(hist)
    future = final_model.make_future_dataframe(periods=horizon, freq="B")
    forecast = final_model.predict(future).tail(horizon)

    return {
        "dates": list(pd.to_datetime(forecast["ds"])),
        "forecast": list(forecast["yhat"].values),
        "lower": list(forecast["yhat_lower"].values),
        "upper": list(forecast["yhat_upper"].values),
        "holdout_metrics": holdout_metrics,
    }


# --------------------------------------------------------------------------- #
# Deep learning: LSTM / GRU
# --------------------------------------------------------------------------- #

def _run_deep_sequence(df, horizon, layer_type="LSTM"):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

    tf.random.set_seed(42)
    close = df["close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    window = 60
    if len(scaled) < window + 30:
        raise ValueError(f"Not enough data to train {layer_type}")

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    RecurrentLayer = LSTM if layer_type == "LSTM" else GRU
    model = Sequential([
        RecurrentLayer(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        RecurrentLayer(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0,
              validation_data=(X_test, y_test) if len(X_test) > 0 else None)

    test_pred_scaled = model.predict(X_test, verbose=0).flatten() if len(X_test) > 0 else np.array([])
    if len(test_pred_scaled) > 0:
        test_pred = scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
        test_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        holdout_metrics = regression_metrics(test_true, test_pred)
        residual_std = float(np.std(test_true - test_pred))
    else:
        holdout_metrics = {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan, "directional_accuracy": np.nan}
        residual_std = float(df["close"].std() * 0.02)

    # Iterative multi-step forecast
    last_window = scaled[-window:].reshape(1, window, 1)
    preds_scaled = []
    for _ in range(horizon):
        next_scaled = model.predict(last_window, verbose=0)[0, 0]
        preds_scaled.append(next_scaled)
        last_window = np.append(last_window[:, 1:, :], [[[next_scaled]]], axis=1)

    forecasts = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    dates = _future_business_days(df["date"].iloc[-1], horizon)
    band = _residual_band(residual_std, horizon)

    return {
        "dates": list(dates),
        "forecast": list(forecasts),
        "lower": list(forecasts - band),
        "upper": list(forecasts + band),
        "holdout_metrics": holdout_metrics,
    }


def run_lstm(df, horizon):
    return _run_deep_sequence(df, horizon, "LSTM")


def run_gru(df, horizon):
    return _run_deep_sequence(df, horizon, "GRU")


# --------------------------------------------------------------------------- #
# Ensemble + Auto model selection
# --------------------------------------------------------------------------- #

MODEL_REGISTRY = {
    "Linear Regression": run_linear_regression,
    "Random Forest": run_random_forest,
    "XGBoost": run_xgboost,
    "LightGBM": run_lightgbm,
    "ARIMA": run_arima,
    "Prophet": run_prophet,
    "LSTM": run_lstm,
    "GRU": run_gru,
}


def run_ensemble(df, horizon, results: dict):
    """Average the forecasts of whichever individual models already
    succeeded (passed in as `results`, a dict of model_name -> result)."""
    valid = [r for r in results.values() if r is not None]
    if not valid:
        raise ValueError("No successful models to build an ensemble from")

    n = min(len(r["forecast"]) for r in valid)
    stacked = np.array([r["forecast"][:n] for r in valid])
    lower_stacked = np.array([r["lower"][:n] for r in valid])
    upper_stacked = np.array([r["upper"][:n] for r in valid])

    avg_rmse = np.nanmean([r["holdout_metrics"].get("rmse", np.nan) for r in valid])
    avg_mae = np.nanmean([r["holdout_metrics"].get("mae", np.nan) for r in valid])
    avg_mape = np.nanmean([r["holdout_metrics"].get("mape", np.nan) for r in valid])
    avg_r2 = np.nanmean([r["holdout_metrics"].get("r2", np.nan) for r in valid])
    avg_dir = np.nanmean([r["holdout_metrics"].get("directional_accuracy", np.nan) for r in valid])

    return {
        "dates": valid[0]["dates"][:n],
        "forecast": list(stacked.mean(axis=0)),
        "lower": list(lower_stacked.min(axis=0)),
        "upper": list(upper_stacked.max(axis=0)),
        "holdout_metrics": {"rmse": avg_rmse, "mae": avg_mae, "mape": avg_mape,
                             "r2": avg_r2, "directional_accuracy": avg_dir},
    }


def pick_best_model(results: dict) -> str:
    """Auto AI Model Selection: choose the model with the lowest holdout RMSE."""
    scored = {name: r["holdout_metrics"].get("rmse", np.inf)
              for name, r in results.items() if r is not None}
    scored = {k: (v if not np.isnan(v) else np.inf) for k, v in scored.items()}
    if not scored:
        return None
    return min(scored, key=scored.get)


def run_selected_models(df, horizon, model_names):
    """Run every model in `model_names` (any of MODEL_REGISTRY keys, plus
    'Ensemble' and 'Auto AI Model Selection'), returning (results, errors)."""
    results, errors = {}, {}
    base_models = [m for m in model_names if m in MODEL_REGISTRY]

    for name in base_models:
        try:
            results[name] = MODEL_REGISTRY[name](df, horizon)
        except Exception as e:
            errors[name] = str(e)
            results[name] = None

    if "Ensemble" in model_names:
        try:
            results["Ensemble"] = run_ensemble(df, horizon, results)
        except Exception as e:
            errors["Ensemble"] = str(e)
            results["Ensemble"] = None

    if "Auto AI Model Selection" in model_names:
        candidates = {k: v for k, v in results.items() if k != "Auto AI Model Selection" and v is not None}
        best = pick_best_model(candidates)
        if best:
            results["Auto AI Model Selection"] = {**candidates[best], "chosen_model": best}
        else:
            errors["Auto AI Model Selection"] = "No underlying model succeeded"
            results["Auto AI Model Selection"] = None

    return results, errors
