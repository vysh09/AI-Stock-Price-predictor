"""Model evaluation metrics and financial risk/return metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan, "directional_accuracy": np.nan}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else np.nan
    r2 = float(r2_score(y_true, y_pred)) if len(set(y_true)) > 1 else np.nan

    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    directional_accuracy = float(np.mean(true_dir == pred_dir) * 100) if len(true_dir) > 0 else np.nan

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "directional_accuracy": directional_accuracy}


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def cagr(start_value: float, end_value: float, years: float) -> float:
    if start_value <= 0 or years <= 0:
        return np.nan
    return (end_value / start_value) ** (1 / years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        return np.nan
    return float(returns.std() * np.sqrt(252))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    return float((excess.mean() / returns.std()) * np.sqrt(252))


def max_drawdown(close: pd.Series) -> float:
    cummax = close.cummax()
    drawdown = (close - cummax) / cummax
    return float(drawdown.min() * 100)


def beta_alpha(stock_returns: pd.Series, bench_returns: pd.Series, risk_free_rate: float = 0.06) -> dict:
    aligned = pd.concat([stock_returns, bench_returns], axis=1).dropna()
    aligned.columns = ["stock", "bench"]
    if len(aligned) < 10 or aligned["bench"].var() == 0:
        return {"beta": np.nan, "alpha": np.nan}
    cov = np.cov(aligned["stock"], aligned["bench"])[0][1]
    beta = cov / aligned["bench"].var()
    daily_rf = risk_free_rate / 252
    expected = daily_rf + beta * (aligned["bench"].mean() - daily_rf)
    alpha = (aligned["stock"].mean() - expected) * 252
    return {"beta": float(beta), "alpha": float(alpha)}


def risk_score(volatility: float, max_dd: float, beta: float) -> float:
    """A 0-100 composite risk score (higher = riskier). Purely heuristic,
    for illustrative comparison between stocks - not a formal risk model."""
    vol_component = min(volatility / 0.6, 1) * 40 if not np.isnan(volatility) else 20
    dd_component = min(abs(max_dd) / 60, 1) * 35 if not np.isnan(max_dd) else 17.5
    beta_component = min(abs(beta) / 2, 1) * 25 if not np.isnan(beta) else 12.5
    return round(vol_component + dd_component + beta_component, 1)
