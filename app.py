"""
AI Stock Predictor - premium, offline, multi-model stock forecasting app.
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_utils import (
    resolve_symbol, fetch_history, fetch_quote_info, fetch_top_movers,
    benchmark_symbol_for, TRENDING_STOCKS, PERIOD_TO_HORIZON_DAYS,
)
from utils.indicators import add_all_indicators, support_resistance
from utils.metrics import (
    daily_returns, cagr, annualized_volatility, sharpe_ratio,
    max_drawdown, beta_alpha, risk_score,
)
from utils.models import run_selected_models
from utils.styles import CUSTOM_CSS, metric_card_html, trend_badge_html

st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

ALL_MODEL_NAMES = [
    "Linear Regression", "Random Forest", "XGBoost", "LightGBM",
    "ARIMA", "Prophet", "LSTM", "GRU", "Ensemble", "Auto AI Model Selection",
]

# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 📈 AI Stock Predictor")
    page = st.radio("Navigate", ["🏠 Home", "🔮 Predict", "⚖️ Compare", "🌐 Market Overview"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Forecasts are probabilistic estimates based on historical patterns and technical indicators. "
               "They are **not** guaranteed future prices and should not be the sole basis for investment decisions.")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def render_hero():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>AI Stock Predictor</h1>
            <p>Predict future stock prices using Artificial Intelligence, Machine Learning, and Deep Learning.
            Search any stock on Yahoo Finance — US, Indian, or global — and get multi-model forecasts,
            technical analysis, and risk metrics in seconds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ticker_strip():
    movers = fetch_top_movers(TRENDING_STOCKS)
    if movers.empty:
        return
    pills = ""
    for _, row in movers.iterrows():
        color = "#2f9e5b" if row["pct_change"] >= 0 else "#c65a45"
        arrow = "▲" if row["pct_change"] >= 0 else "▼"
        pills += (f'<div class="ticker-pill">{row["symbol"]} &nbsp; '
                  f'<span style="color:{color}">{arrow} {row["pct_change"]:.2f}%</span></div>')
    st.markdown(f'<div class="ticker-strip">{pills}</div>', unsafe_allow_html=True)


def build_candlestick_fig(df, title):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.03,
        subplot_titles=(title, "RSI (14)", "MACD"),
    )
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price", increasing_line_color="#2f9e5b", decreasing_line_color="#c65a45",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma_20"], name="SMA 20", line=dict(color="#F8C685", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma_50"], name="SMA 50", line=dict(color="#544A84", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["bb_upper"], name="BB Upper", line=dict(color="#c9c2e8", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["bb_lower"], name="BB Lower", line=dict(color="#c9c2e8", width=1, dash="dot"), fill="tonexty"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["date"], y=df["rsi_14"], name="RSI", line=dict(color="#37315F", width=1.4)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#c65a45", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#2f9e5b", row=2, col=1)

    fig.add_trace(go.Bar(x=df["date"], y=df["macd_hist"], name="MACD Hist", marker_color="#F6A696"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["macd"], name="MACD", line=dict(color="#37315F", width=1.3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], name="Signal", line=dict(color="#F8C685", width=1.3)), row=3, col=1)

    fig.update_layout(height=760, template="plotly_white", showlegend=True,
                       margin=dict(l=10, r=10, t=40, b=10), xaxis_rangeslider_visible=False)
    return fig


def build_forecast_fig(hist_df, result, model_name):
    fig = go.Figure()
    tail = hist_df.tail(120)
    fig.add_trace(go.Scatter(x=tail["date"], y=tail["close"], name="Historical Close",
                              line=dict(color="#37315F", width=2)))
    fig.add_trace(go.Scatter(x=result["dates"], y=result["forecast"], name=f"{model_name} Forecast",
                              line=dict(color="#F8C685", width=2.5)))
    fig.add_trace(go.Scatter(x=result["dates"], y=result["upper"], name="Upper CI",
                              line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=result["dates"], y=result["lower"], name="Confidence Interval",
                              line=dict(width=0), fill="tonexty", fillcolor="rgba(84,74,132,0.15)"))
    fig.update_layout(height=440, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10),
                       legend=dict(orientation="h", y=1.1))
    return fig


def generate_ai_explanation(symbol, latest, trend, pct_return):
    reasons = []
    if latest["close"] > latest["sma_50"] > latest["sma_200"] if not np.isnan(latest.get("sma_200", np.nan)) else latest["close"] > latest["sma_50"]:
        reasons.append("price is trading above both its 20-day and 50-day moving averages, a classic uptrend signal")
    elif latest["close"] < latest["sma_50"]:
        reasons.append("price is trading below its 50-day moving average, suggesting near-term weakness")

    if latest["rsi_14"] >= 70:
        reasons.append(f"RSI is at {latest['rsi_14']:.0f}, in overbought territory, which can precede a pullback")
    elif latest["rsi_14"] <= 30:
        reasons.append(f"RSI is at {latest['rsi_14']:.0f}, in oversold territory, which can precede a bounce")
    else:
        reasons.append(f"RSI is neutral at {latest['rsi_14']:.0f}")

    if latest["macd"] > latest["macd_signal"]:
        reasons.append("MACD is above its signal line, indicating positive momentum")
    else:
        reasons.append("MACD is below its signal line, indicating fading momentum")

    reason_text = "; ".join(reasons)
    return (f"The model projects a **{trend.lower()}** outlook for **{symbol}**, with an expected "
            f"return of **{pct_return:+.2f}%** over the selected horizon. This is based on current "
            f"technical conditions: {reason_text}. Historical price action, momentum, and volume "
            f"patterns were used as model inputs alongside the selected forecasting algorithm(s).")


def confidence_score_from_metrics(metrics: dict) -> float:
    r2 = metrics.get("r2", np.nan)
    dir_acc = metrics.get("directional_accuracy", np.nan)
    r2_component = max(min(r2, 1), 0) * 100 if not np.isnan(r2) else 50
    dir_component = dir_acc if not np.isnan(dir_acc) else 50
    return round((r2_component * 0.5 + dir_component * 0.5), 1)


# --------------------------------------------------------------------------- #
# HOME PAGE
# --------------------------------------------------------------------------- #
if page == "🏠 Home":
    render_hero()
    st.markdown('<div class="section-title">Trending Now</div>', unsafe_allow_html=True)
    render_ticker_strip()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="section-title">Quick Search</div>', unsafe_allow_html=True)
        quick_symbol = st.text_input("Enter a stock symbol or company name",
                                      placeholder="e.g. AAPL, Tesla, RELIANCE.NS, TCS.NS")
        if st.button("Predict this stock →"):
            st.session_state["prefill_symbol"] = quick_symbol
            st.info("Symbol saved — switch to the **🔮 Predict** tab in the sidebar to run the forecast.")

    with col2:
        st.markdown('<div class="section-title">Examples</div>', unsafe_allow_html=True)
        st.markdown(
            "- 🇺🇸 `AAPL`, `MSFT`, `NVDA`, `TSLA`, `GOOG`, `AMZN`\n"
            "- 🇮🇳 `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`\n"
            "- 🌐 Any valid Yahoo Finance ticker works"
        )

    st.markdown('<div class="section-title">Top Gainers & Losers (Trending List)</div>', unsafe_allow_html=True)
    movers = fetch_top_movers(TRENDING_STOCKS)
    if not movers.empty:
        gcol, lcol = st.columns(2)
        gainers = movers.sort_values("pct_change", ascending=False).head(5)
        losers = movers.sort_values("pct_change", ascending=True).head(5)
        with gcol:
            st.markdown("**📈 Top Gainers**")
            st.dataframe(gainers, hide_index=True, use_container_width=True)
        with lcol:
            st.markdown("**📉 Top Losers**")
            st.dataframe(losers, hide_index=True, use_container_width=True)

    st.info("📰 Live financial news & AI sentiment scoring, portfolio dashboard, and user accounts are "
            "planned for a later phase of this project — this build focuses on the core AI prediction engine.")


# --------------------------------------------------------------------------- #
# PREDICT PAGE
# --------------------------------------------------------------------------- #
elif page == "🔮 Predict":
    st.markdown('<div class="section-title">Run a Prediction</div>', unsafe_allow_html=True)

    default_symbol = st.session_state.get("prefill_symbol", "AAPL")
    c1, c2, c3 = st.columns([2, 1.2, 2])
    with c1:
        symbol_input = st.text_input("Stock symbol or company name", value=default_symbol)
    with c2:
        horizon_label = st.selectbox("Prediction Duration", list(PERIOD_TO_HORIZON_DAYS.keys()), index=1)
    with c3:
        chosen_models = st.multiselect("Models to run", ALL_MODEL_NAMES,
                                        default=["Prophet", "LSTM", "XGBoost", "Auto AI Model Selection", "Ensemble"])

    run = st.button("🚀 Generate Prediction", type="primary")

    if run:
        symbol = resolve_symbol(symbol_input)
        if not symbol:
            st.error("Please enter a valid stock symbol or company name.")
            st.stop()

        with st.spinner(f"Downloading historical data for {symbol}..."):
            hist = fetch_history(symbol, years=10)

        if hist.empty:
            st.error(f"Couldn't find data for '{symbol}' on Yahoo Finance. Check the symbol and try again "
                     f"(e.g. add `.NS` for NSE-listed Indian stocks).")
            st.stop()

        hist_ind = add_all_indicators(hist)
        quote = fetch_quote_info(symbol)
        current_price = float(hist_ind["close"].iloc[-1])
        horizon_days = min(PERIOD_TO_HORIZON_DAYS[horizon_label], 252 * 10)
        # Cap iterative-forecast compute cost for very long horizons on daily models
        forecast_steps = min(horizon_days, 756)  # ~3 trading years max iterative steps

        st.success(f"Loaded {len(hist_ind):,} trading days of history for **{symbol}**.")

        with st.spinner("Training models and generating forecasts... this can take a minute for deep learning models."):
            results, errors = run_selected_models(hist_ind, forecast_steps, chosen_models)

        if errors:
            with st.expander("⚠️ Some models had issues"):
                for name, err in errors.items():
                    st.write(f"**{name}**: {err}")

        successful = {k: v for k, v in results.items() if v is not None}
        if not successful:
            st.error("None of the selected models could be trained on this data. Try a different symbol or model set.")
            st.stop()

        # Prefer Auto AI Model Selection / Ensemble as the headline result if present
        headline_name = "Auto AI Model Selection" if results.get("Auto AI Model Selection") else \
            ("Ensemble" if results.get("Ensemble") else next(iter(successful)))
        headline = successful[headline_name]

        predicted_price = float(headline["forecast"][-1])
        pct_return = (predicted_price - current_price) / current_price * 100
        trend = "Bullish" if pct_return > 2 else ("Bearish" if pct_return < -2 else "Neutral")
        confidence = confidence_score_from_metrics(headline["holdout_metrics"])

        # Financial metrics
        rets = daily_returns(hist_ind["close"])
        vol = annualized_volatility(rets)
        shrp = sharpe_ratio(rets)
        mdd = max_drawdown(hist_ind["close"])
        years_elapsed = len(hist_ind) / 252
        historical_cagr = cagr(hist_ind["close"].iloc[0], hist_ind["close"].iloc[-1], years_elapsed)
        forecast_years = forecast_steps / 252
        forecast_cagr = cagr(current_price, predicted_price, max(forecast_years, 1 / 252))

        bench_symbol = benchmark_symbol_for(symbol)
        bench_hist = fetch_history(bench_symbol, years=5)
        if not bench_hist.empty:
            bench_rets = daily_returns(bench_hist["close"])
            ba = beta_alpha(rets.tail(len(bench_rets)), bench_rets.tail(len(rets)))
        else:
            ba = {"beta": np.nan, "alpha": np.nan}
        rscore = risk_score(vol, mdd, ba["beta"])

        # --- Headline metric cards ---
        st.markdown(f'<div class="section-title">{symbol} — Forecast Summary ({headline_name})</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(metric_card_html("Current Price", f"{current_price:,.2f}"), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card_html(f"Predicted Price ({horizon_label})", f"{predicted_price:,.2f}"), unsafe_allow_html=True)
        with m3:
            st.markdown(metric_card_html("Expected Return", f"{pct_return:+.2f}%"), unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="label">Trend</div><div class="value">{trend_badge_html(trend)}</div></div>', unsafe_allow_html=True)

        m5, m6, m7, m8 = st.columns(4)
        with m5:
            st.markdown(metric_card_html("Confidence Score", f"{confidence:.0f}/100"), unsafe_allow_html=True)
        with m6:
            st.markdown(metric_card_html("Expected CAGR", f"{forecast_cagr*100:,.2f}%" if not np.isnan(forecast_cagr) else "N/A"), unsafe_allow_html=True)
        with m7:
            st.markdown(metric_card_html("Volatility (ann.)", f"{vol*100:,.2f}%" if not np.isnan(vol) else "N/A"), unsafe_allow_html=True)
        with m8:
            st.markdown(metric_card_html("Risk Score", f"{rscore}/100"), unsafe_allow_html=True)

        m9, m10, m11, m12 = st.columns(4)
        with m9:
            st.markdown(metric_card_html("Max Drawdown", f"{mdd:,.2f}%"), unsafe_allow_html=True)
        with m10:
            st.markdown(metric_card_html("Sharpe Ratio", f"{shrp:,.2f}" if not np.isnan(shrp) else "N/A"), unsafe_allow_html=True)
        with m11:
            st.markdown(metric_card_html("Beta", f"{ba['beta']:.2f}" if not np.isnan(ba['beta']) else "N/A"), unsafe_allow_html=True)
        with m12:
            st.markdown(metric_card_html("Alpha (ann.)", f"{ba['alpha']*100:.2f}%" if not np.isnan(ba['alpha']) else "N/A"), unsafe_allow_html=True)

        # --- AI explanation ---
        st.markdown('<div class="section-title">AI-Generated Explanation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card">{generate_ai_explanation(symbol, hist_ind.iloc[-1], trend, pct_return)}</div>',
                    unsafe_allow_html=True)

        # --- Charts ---
        st.markdown('<div class="section-title">Historical Price & Technical Indicators</div>', unsafe_allow_html=True)
        st.plotly_chart(build_candlestick_fig(hist_ind, f"{symbol} Price"), use_container_width=True)

        sr = support_resistance(hist_ind)
        sr_cols = st.columns(5)
        for col, (label, val) in zip(sr_cols, sr.items()):
            with col:
                st.markdown(metric_card_html(label.replace("_", " ").title(), f"{val:,.2f}"), unsafe_allow_html=True)

        st.markdown('<div class="section-title">Future Forecast</div>', unsafe_allow_html=True)
        st.plotly_chart(build_forecast_fig(hist_ind, headline, headline_name), use_container_width=True)

        # --- Model comparison table ---
        st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
        rows = []
        for name, r in results.items():
            if r is None:
                continue
            met = r["holdout_metrics"]
            rows.append({
                "Model": name + (f" (chose {r['chosen_model']})" if r.get("chosen_model") else ""),
                "Forecast Price": round(r["forecast"][-1], 2),
                "RMSE": round(met.get("rmse", np.nan), 3) if not np.isnan(met.get("rmse", np.nan)) else None,
                "MAE": round(met.get("mae", np.nan), 3) if not np.isnan(met.get("mae", np.nan)) else None,
                "MAPE %": round(met.get("mape", np.nan), 2) if not np.isnan(met.get("mape", np.nan)) else None,
                "R²": round(met.get("r2", np.nan), 3) if not np.isnan(met.get("r2", np.nan)) else None,
                "Directional Acc %": round(met.get("directional_accuracy", np.nan), 1) if not np.isnan(met.get("directional_accuracy", np.nan)) else None,
            })
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        # --- Export ---
        st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
        export_df = pd.DataFrame({
            "date": headline["dates"], "forecast": headline["forecast"],
            "lower_ci": headline["lower"], "upper_ci": headline["upper"],
        })
        st.download_button("⬇️ Download Forecast as CSV", export_df.to_csv(index=False),
                            file_name=f"{symbol}_forecast.csv", mime="text/csv")


# --------------------------------------------------------------------------- #
# COMPARE PAGE
# --------------------------------------------------------------------------- #
elif page == "⚖️ Compare":
    st.markdown('<div class="section-title">Compare Multiple Stocks</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        symbols_input = st.text_input("Enter symbols separated by commas", value="AAPL, MSFT, TCS.NS")
    with c2:
        horizon_label = st.selectbox("Duration", list(PERIOD_TO_HORIZON_DAYS.keys()), index=1, key="compare_horizon")

    if st.button("Compare"):
        symbols = [resolve_symbol(s) for s in symbols_input.split(",") if s.strip()]
        horizon_days = min(PERIOD_TO_HORIZON_DAYS[horizon_label], 252 * 3)

        fig = go.Figure()
        summary_rows = []
        for sym in symbols:
            hist = fetch_history(sym, years=5)
            if hist.empty:
                st.warning(f"No data for {sym}, skipping.")
                continue
            hist_ind = add_all_indicators(hist)
            normalized = hist_ind["close"] / hist_ind["close"].iloc[0] * 100
            fig.add_trace(go.Scatter(x=hist_ind["date"], y=normalized, name=sym))

            with st.spinner(f"Forecasting {sym}..."):
                results, _ = run_selected_models(hist_ind, horizon_days, ["Auto AI Model Selection", "Prophet", "XGBoost"])
            r = results.get("Auto AI Model Selection") or next((v for v in results.values() if v), None)
            if r:
                current = hist_ind["close"].iloc[-1]
                predicted = r["forecast"][-1]
                pct = (predicted - current) / current * 100
                rets = daily_returns(hist_ind["close"])
                summary_rows.append({
                    "Symbol": sym, "Current Price": round(current, 2), "Predicted Price": round(predicted, 2),
                    "Expected Return %": round(pct, 2),
                    "Trend": "Bullish" if pct > 2 else ("Bearish" if pct < -2 else "Neutral"),
                    "Volatility %": round(annualized_volatility(rets) * 100, 2),
                    "Sharpe": round(sharpe_ratio(rets), 2),
                })

        fig.update_layout(title="Normalized Price Comparison (Base = 100)", height=460, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        if summary_rows:
            st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- #
# MARKET OVERVIEW PAGE
# --------------------------------------------------------------------------- #
elif page == "🌐 Market Overview":
    st.markdown('<div class="section-title">Market Snapshot</div>', unsafe_allow_html=True)
    movers = fetch_top_movers(TRENDING_STOCKS)
    if not movers.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📈 Top Gainers**")
            st.dataframe(movers.sort_values("pct_change", ascending=False).head(5), hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**📉 Top Losers**")
            st.dataframe(movers.sort_values("pct_change", ascending=True).head(5), hide_index=True, use_container_width=True)

        fig = go.Figure(go.Bar(
            x=movers["symbol"], y=movers["pct_change"],
            marker_color=["#2f9e5b" if v >= 0 else "#c65a45" for v in movers["pct_change"]],
        ))
        fig.update_layout(title="Daily % Change — Trending Watchlist", height=380, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.info("🗓️ Sector heatmap, Fear & Greed Index, earnings/dividend calendars, and live news sentiment "
            "are planned for a later phase — they require paid data APIs beyond Yahoo Finance.")
