
"""
Put/Call-Spread Backtest – separate Streamlit-App
==================================================
Testet die Spread-Strategie (INDEX_PC - EQUITY_PC) gegen Buy-and-Hold SPY.

Wichtig: Alle angezeigten Zahlen sind Berechnungsergebnisse aus den geladenen
Daten. Keine illustrativen Werte. Die Aussagekraft ist begrenzt durch die
verfügbare Datenhistorie (~10 Jahre bei Equibles) und die Modellierung.
"""

import os
import pickle
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ============================================================
# KONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Backtest Put/Call-Spread",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Backtest: Put/Call-Spread-Indikator")
st.caption(
    "Alle angezeigten Kennzahlen sind echte Berechnungsergebnisse aus den geladenen "
    "Daten – keine illustrativen Werte. Aussagekraft begrenzt durch ~10 Jahre "
    "Datenhistorie und Modellannahmen (siehe Tab 'Methodik & Grenzen')."
)

CACHE_FILE = "backtest_pc_cache.pkl"
CACHE_TTL_HOURS = 24
EQUIBLES_BASE = "https://api.equibles.com/v1/market/put-call-ratios"


# ============================================================
# DATEN LADEN
# ============================================================
def _fetch_equibles_series(series_type, api_key, start_date, end_date):
    headers = {"Authorization": f"Bearer {api_key}"}
    all_rows = []
    offset = 0
    page_limit = 500
    for _ in range(30):
        params = {
            "type": series_type,
            "limit": page_limit,
            "offset": offset,
            "startDate": start_date,
            "endDate": end_date,
        }
        resp = requests.get(EQUIBLES_BASE, headers=headers, params=params, timeout=30)
        if resp.status_code == 401:
            raise RuntimeError("Equibles-API-Key ungültig.")
        if resp.status_code == 429:
            raise RuntimeError("RATE_LIMIT")
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_limit:
            break
        if not payload.get("meta", {}).get("hasMore", True):
            break
        offset += page_limit
        time.sleep(0.25)
    if not all_rows:
        raise RuntimeError(f"Keine Daten für '{series_type}'.")
    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"date": "DATE", "putCallRatio": "RATIO"})
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["RATIO"] = pd.to_numeric(df["RATIO"], errors="coerce")
    return df[["DATE", "RATIO"]].dropna().sort_values("DATE").reset_index(drop=True)


def _load_cache():
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "rb") as f:
            payload = pickle.load(f)
        age = (datetime.now() - payload["timestamp"]).total_seconds() / 3600
        if age > CACHE_TTL_HOURS:
            return None
        return payload["df"]
    except Exception:
        return None


def _save_cache(df):
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({"df": df, "timestamp": datetime.now()}, f)
    except Exception:
        pass


def _fetch_pc_data(api_key, start_date, end_date):
    df_index = _fetch_equibles_series("Index", api_key, start_date, end_date)
    df_equity = _fetch_equibles_series("Equity", api_key, start_date, end_date)
    df_index = df_index.rename(columns={"RATIO": "INDEX_PC"})
    df_equity = df_equity.rename(columns={"RATIO": "EQUITY_PC"})
    return pd.merge(df_index, df_equity, on="DATE", how="inner")


def _load_pc_cached(api_key, start_date, end_date):
    cached = _load_cache()
    if cached is not None:
        return cached, "cache"
    fresh = _fetch_pc_data(api_key, start_date, end_date)
    _save_cache(fresh)
    return fresh, "api"


@st.cache_data(ttl=86400, show_spinner=False)
def load_pc_data(api_key, start_date, end_date):
    return _load_pc_cached(api_key, start_date, end_date)


@st.cache_data(ttl=86400, show_spinner=False)
def load_spy_data():
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance ist nicht installiert.")
    hist = yf.Ticker("SPY").history(period="20y", interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        raise RuntimeError("yfinance lieferte keine SPY-Daten.")
    out = hist[["Close"]].reset_index()
    out.rename(columns={out.columns[0]: "DATE", "Close": "SPY_CLOSE"}, inplace=True)
    out["DATE"] = pd.to_datetime(out["DATE"]).dt.tz_localize(None)
    return out.dropna().sort_values("DATE").reset_index(drop=True)


# ============================================================
# SIGNAL-GENERIERUNG
# ============================================================
def generate_signal(spread, strategy, **params):
    """Erzeugt Signal-Reihe: +1 = long, 0 = flat, -1 = short.

    WICHTIG: Signal wird noch NICHT verschoben. Die Verschiebung passiert
    im Backtest (signal.shift(1)), um Look-ahead-Bias zu vermeiden.
    """
    if strategy == "sma_cross":
        sma = spread.rolling(params["sma_period"], min_periods=params["sma_period"]).mean()
        signal = (spread < sma).astype(float)

    elif strategy == "zscore":
        window = params["zscore_window"]
        rolling_mean = spread.rolling(window, min_periods=window).mean()
        rolling_std = spread.rolling(window, min_periods=window).std()
        z = (spread - rolling_mean) / rolling_std
        signal = (z < -params["z_threshold"]).astype(float)
        # Exit, wenn z >= 0
        signal = signal.where(z < 0, 0)

    elif strategy == "zscore_ls":
        window = params["zscore_window"]
        rolling_mean = spread.rolling(window, min_periods=window).mean()
        rolling_std = spread.rolling(window, min_periods=window).std()
        z = (spread - rolling_mean) / rolling_std
        signal = pd.Series(0.0, index=spread.index)
        signal[z < -params["z_threshold"]] = 1.0
        signal[z > params["z_threshold"]] = -1.0

    elif strategy == "buy_hold":
        signal = pd.Series(1.0, index=spread.index)

    else:
        raise ValueError(f"Unbekannte Strategie: {strategy}")

    return signal.fillna(0)


# ============================================================
# BACKTEST-ENGINE
# ============================================================
def run_backtest(underlying_returns, signal, cost_bps=5):
    """Führt Backtest durch.

    underlying_returns: Series mit Tagesrenditen des Underlyings (SPY)
    signal:             Series mit gewünschter Position (+1/0/-1) am Tag t
    cost_bps:           Kosten pro Trade in Basispunkten
    """
    # Position: Signal von Tag t gilt ab Tag t+1
    position = signal.shift(1).fillna(0)

    # Strategie-Rendite: Position * Underlying-Rendite - Kosten bei Positionsänderung
    cost_rate = cost_bps / 10000.0
    strat_returns = position * underlying_returns - position.diff().abs().fillna(0) * cost_rate

    equity = (1 + strat_returns).cumprod()
    return {
        "returns": strat_returns,
        "equity": equity,
        "position": position,
        "num_trades": int((position.diff().abs() > 0).sum()),
    }


# ============================================================
# METRIKEN
# ============================================================
def compute_metrics(returns, equity, freq=252):
    if len(returns) == 0 or len(equity) == 0:
        return {k: np.nan for k in [
            "total_return", "cagr", "vol", "sharpe", "sortino",
            "max_dd", "calmar", "win_rate", "exposure",
        ]}

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = len(returns) / freq
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(freq)
    sharpe = (returns.mean() / returns.std() * np.sqrt(freq)) if returns.std() > 0 else 0

    downside = returns[returns < 0]
    downside_std = downside.std()
    sortino = (returns.mean() / downside_std * np.sqrt(freq)) if downside_std > 0 else 0

    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    win_rate = (returns > 0).mean()

    return {
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
    }


def drawdown_series(equity):
    cummax = equity.cummax()
    return (equity - cummax) / cummax


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward(df, strategy_name, param_grid, cost_bps, train_years=3, test_years=1):
    """Rollierender Walk-Forward-Test.

    Auf dem Trainingsfenster wird der Parameter mit der besten Sharpe Ratio
    gewählt. Dieser wird auf das folgende Testfenster angewendet.
    """
    if "RETURNS" not in df.columns:
        raise ValueError("df muss Spalte 'RETURNS' enthalten.")

    start = df["DATE"].min()
    end = df["DATE"].max()

    train_start = start
    oos_returns = []
    oos_params = []
    oos_dates = []

    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > end:
            break

        train_df = df[(df["DATE"] >= train_start) & (df["DATE"] < train_end)].reset_index(drop=True)
        test_df = df[(df["DATE"] >= train_end) & (df["DATE"] < test_end)].reset_index(drop=True)

        if len(train_df) < 200 or len(test_df) < 50:
            train_start = train_start + pd.DateOffset(years=test_years)
            continue

        # Optimierung auf Trainingsfenster
        best_sharpe = -np.inf
        best_params = None
        for params in param_grid:
            sig = generate_signal(train_df["SPREAD"], strategy_name, **params)
            res = run_backtest(train_df["RETURNS"], sig, cost_bps)
            m = compute_metrics(res["returns"], res["equity"])
            if not np.isnan(m["sharpe"]) and m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_params = params

        if best_params is None:
            train_start = train_start + pd.DateOffset(years=test_years)
            continue

        # Anwendung auf Testfenster
        sig_test = generate_signal(test_df["SPREAD"], strategy_name, **best_params)
        res_test = run_backtest(test_df["RETURNS"], sig_test, cost_bps)

        oos_returns.append(res_test["returns"].values)
        oos_params.append(best_params)
        oos_dates.append(test_df["DATE"].values)

        train_start = train_start + pd.DateOffset(years=test_years)

    if not oos_returns:
        return None

    all_returns = np.concatenate(oos_returns)
    all_dates = np.concatenate(oos_dates)
    oos_equity = (1 + pd.Series(all_returns, index=all_dates)).cumprod()

    return {
        "dates": pd.to_datetime(all_dates),
        "returns": pd.Series(all_returns, index=pd.to_datetime(all_dates)),
        "equity": oos_equity,
        "params": oos_params,
    }


# ============================================================
# MONTE CARLO
# ============================================================
def monte_carlo_block_bootstrap(returns, block_size=20, n_sim=2000, horizon=None, seed=42):
    """Block-Bootstrap auf Tagesrenditen zur Schätzung der Ergebnisverteilung."""
    rng = np.random.default_rng(seed)
    arr = returns.dropna().values

    if len(arr) < block_size * 2:
        return None

    if horizon is None:
        horizon = len(arr)

    n_blocks = int(np.ceil(horizon / block_size))
    results = []

    for _ in range(n_sim):
        starts = rng.integers(0, len(arr) - block_size, n_blocks)
        sim = np.concatenate([arr[s:s + block_size] for s in starts])[:horizon]
        cum = np.cumprod(1 + sim)
        final = cum[-1]
        max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
        results.append({"final": final, "max_dd": max_dd})

    return pd.DataFrame(results)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Einstellungen")

api_key = st.secrets.get("EQUIBLES_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("Equibles API-Key", type="password")
else:
    st.sidebar.success("✅ API-Key aus Secrets geladen")

st.sidebar.divider()
st.sidebar.subheader("📅 Zeitraum")
years_back = st.sidebar.slider("Jahre zurück:", 2, 12, 10)

st.sidebar.divider()
st.sidebar.subheader("🎯 Strategie")

STRATEGIES = {
    "SMA-Cross (Long/Flat)": "sma_cross",
    "Z-Score (Long/Flat)": "zscore",
    "Z-Score (Long/Short)": "zscore_ls",
    "Buy & Hold SPY": "buy_hold",
}
strategy_label = st.sidebar.selectbox("Auswahl:", list(STRATEGIES.keys()))
strategy_name = STRATEGIES[strategy_label]

# Parameter abhängig von Strategie
if strategy_name == "sma_cross":
    sma_period = st.sidebar.slider("SMA-Periode (Tage):", 5, 200, 20)
    strategy_params = {"sma_period": sma_period}
elif strategy_name in ("zscore", "zscore_ls"):
    zscore_window = st.sidebar.slider("Z-Score Fenster (Tage):", 30, 250, 60)
    z_threshold = st.sidebar.slider("Z-Score Schwelle:", 0.5, 2.5, 1.0, step=0.1)
    strategy_params = {"zscore_window": zscore_window, "z_threshold": z_threshold}
else:
    strategy_params = {}

st.sidebar.divider()
st.sidebar.subheader("💰 Kosten")
cost_bps = st.sidebar.slider(
    "Transaktionskosten (Basispunkte pro Trade):",
    0, 50, 5, step=1,
    help="5 bp ≈ 0,05 % – realistisch für SPY bei einem Broker wie IBKR."
)

st.sidebar.divider()
st.sidebar.subheader("🔄 Walk-Forward")
wf_train_years = st.sidebar.slider("Trainingsfenster (Jahre):", 2, 5, 3)
wf_test_years = st.sidebar.slider("Testfenster (Jahre):", 1, 3, 1)

if st.sidebar.button("🔄 Cache leeren & neu laden"):
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
        except Exception:
            pass
    st.cache_data.clear()
    st.rerun()


# ============================================================
# DATEN LADEN & VORBEREITEN
# ============================================================
if not api_key and strategy_name != "buy_hold":
    st.info("⬅️ Bitte einen Equibles-API-Key eingeben (oder 'Buy & Hold' wählen).")
    st.stop()

end_dt = datetime.now()
start_dt = end_dt - timedelta(days=int(years_back * 365.25) + 300)

try:
    if strategy_name == "buy_hold":
        df_pc = None
        data_source = "—"
    else:
        with st.spinner("Lade Put/Call-Ratio-Daten..."):
            df_pc, data_source = load_pc_data(
                api_key,
                start_dt.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
            )
except Exception as e:
    st.error(f"❌ Fehler beim Laden der Put/Call-Daten: {e}")
    st.stop()

with st.spinner("Lade SPY-Daten..."):
    df_spy = load_spy_data()

# Zusammenführen
if df_pc is not None:
    df_pc = df_pc.sort_values("DATE").reset_index(drop=True)
    df_pc["SPREAD"] = df_pc["INDEX_PC"] - df_pc["EQUITY_PC"]
    merged = pd.merge(df_pc, df_spy, on="DATE", how="inner")
else:
    merged = df_spy.copy()
    merged["SPREAD"] = 0.0

merged = merged.sort_values("DATE").reset_index(drop=True)

# Auf gewünschten Zeitraum beschränken
cutoff = end_dt - timedelta(days=int(years_back * 365.25))
merged = merged[merged["DATE"] >= cutoff].reset_index(drop=True)

# Tagesrenditen des Underlyings
merged["RETURNS"] = merged["SPY_CLOSE"].pct_change().fillna(0)

if len(merged) < 100:
    st.error("Zu wenig Daten für einen sinnvollen Backtest.")
    st.stop()

# Datenquelle-Hinweis
if data_source == "cache":
    st.success(f"✅ {len(merged)} Handelstage geladen – Daten aus Cache.")
elif data_source == "api":
    st.info(f"🔄 {len(merged)} Handelstage geladen – Daten frisch von Equibles-API.")

st.caption(
    f"Datenzeitraum: {merged['DATE'].min().date()} bis {merged['DATE'].max().date()} "
    f"({len(merged)} Handelstage, ≈ {len(merged)/252:.1f} Jahre)"
)


# ============================================================
# BACKTEST DURCHFÜHREN
# ============================================================
signal = generate_signal(merged["SPREAD"], strategy_name, **strategy_params)
bt = run_backtest(merged["RETURNS"], signal, cost_bps=cost_bps)
bench_equity = (1 + merged["RETURNS"]).cumprod()

metrics_strat = compute_metrics(bt["returns"], bt["equity"])
metrics_bench = compute_metrics(merged["RETURNS"], bench_equity)
metrics_strat["exposure"] = (bt["position"] != 0).mean()
metrics_strat["num_trades"] = bt["num_trades"]


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Übersicht",
    "🔍 Signal-Details",
    "🔄 Walk-Forward",
    "🎲 Monte Carlo",
    "📝 Methodik & Grenzen",
])


# ---------- TAB 1: ÜBERSICHT ----------
with tab1:
    st.subheader("Equity-Kurve: Strategie vs. Buy & Hold")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged["DATE"], y=bt["equity"],
        name=f"Strategie: {strategy_label}",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=merged["DATE"], y=bench_equity,
        name="Benchmark: Buy & Hold SPY",
        line=dict(color="#888", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        height=420, template="plotly_white", hovermode="x unified",
        yaxis_title="Kumulierter Wert (Start = 1)",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Kennzahlen im Vergleich")
    col1, col2 = st.columns(2)

    def fmt_pct(x):
        return f"{x*100:.2f} %" if pd.notna(x) else "n/a"

    def fmt_num(x):
        return f"{x:.2f}" if pd.notna(x) else "n/a"

    with col1:
        st.markdown("**Strategie**")
        st.metric("Gesamtrendite", fmt_pct(metrics_strat["total_return"]))
        st.metric("CAGR", fmt_pct(metrics_strat["cagr"]))
        st.metric("Annualisierte Volatilität", fmt_pct(metrics_strat["vol"]))
        st.metric("Sharpe Ratio", fmt_num(metrics_strat["sharpe"]))
        st.metric("Sortino Ratio", fmt_num(metrics_strat["sortino"]))
        st.metric("Max Drawdown", fmt_pct(metrics_strat["max_dd"]))
        st.metric("Calmar Ratio", fmt_num(metrics_strat["calmar"]))
        st.metric("Trefferquote (Tage)", fmt_pct(metrics_strat["win_rate"]))
        st.metric("Exposure (Markt investiert)", fmt_pct(metrics_strat["exposure"]))
        st.metric("Anzahl Trades", f"{metrics_strat['num_trades']}")

    with col2:
        st.markdown("**Buy & Hold SPY**")
        st.metric("Gesamtrendite", fmt_pct(metrics_bench["total_return"]))
        st.metric("CAGR", fmt_pct(metrics_bench["cagr"]))
        st.metric("Annualisierte Volatilität", fmt_pct(metrics_bench["vol"]))
        st.metric("Sharpe Ratio", fmt_num(metrics_bench["sharpe"]))
        st.metric("Sortino Ratio", fmt_num(metrics_bench["sortino"]))
        st.metric("Max Drawdown", fmt_pct(metrics_bench["max_dd"]))
        st.metric("Calmar Ratio", fmt_num(metrics_bench["calmar"]))
        st.metric("Trefferquote (Tage)", fmt_pct(metrics_bench["win_rate"]))

    st.caption(
        "⚠️ Achtung: Selbst wenn die Strategie die Benchmark in dieser Historie schlägt, "
        "ist das noch kein Beweis für einen echten Edge. Siehe Tab 'Methodik & Grenzen'."
    )


# ---------- TAB 2: SIGNAL-DETAILS ----------
with tab2:
    st.subheader("Spread-Indikator und Position")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=(
            "Put/Call-Spread und SMA",
            "Position (nach 1-Tag-Verzögerung)",
            "Drawdown",
        ),
    )

    # Spread und SMA
    fig.add_trace(go.Scatter(
        x=merged["DATE"], y=merged["SPREAD"],
        name="Spread", line=dict(color="darkcyan", width=1.5),
    ), row=1, col=1)
    if strategy_name == "sma_cross":
        sma_series = merged["SPREAD"].rolling(strategy_params["sma_period"]).mean()
        fig.add_trace(go.Scatter(
            x=merged["DATE"], y=sma_series,
            name=f"SMA {strategy_params['sma_period']}",
            line=dict(color="orange", width=2),
        ), row=1, col=1)

    # Position
    fig.add_trace(go.Scatter(
        x=merged["DATE"], y=bt["position"],
        name="Position", line=dict(color="#1f77b4", width=1.2, shape="hv"),
    ), row=2, col=1)

    # Drawdown der Strategie
    dd = drawdown_series(bt["equity"])
    fig.add_trace(go.Scatter(
        x=merged["DATE"], y=dd,
        name="Drawdown", fill="tozeroy",
        line=dict(color="crimson", width=1),
    ), row=3, col=1)

    fig.update_layout(
        height=800, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- TAB 3: WALK-FORWARD ----------
with tab3:
    st.subheader("Walk-Forward-Analyse")
    st.caption(
        f"Rollierender Test: {wf_train_years} Jahre Training (Parameterwahl), "
        f"dann {wf_test_years} Jahr(e) Out-of-Sample. Das deckt Overfitting auf."
    )

    if strategy_name == "sma_cross":
        param_grid = [{"sma_period": p} for p in [10, 20, 30, 50, 75, 100]]
    elif strategy_name == "zscore":
        param_grid = (
            [{"zscore_window": w, "z_threshold": t}
             for w in [40, 60, 90] for t in [0.75, 1.0, 1.25]]
        )
    elif strategy_name == "zscore_ls":
        param_grid = (
            [{"zscore_window": w, "z_threshold": t}
             for w in [40, 60, 90] for t in [0.75, 1.0, 1.25]]
        )
    else:
        param_grid = [{}]

    with st.spinner("Berechne Walk-Forward..."):
        wf = walk_forward(
            merged, strategy_name, param_grid, cost_bps,
            train_years=wf_train_years, test_years=wf_test_years,
        )

    if wf is None:
        st.warning(
            "Walk-Forward konnte nicht berechnet werden – zu wenig Daten für "
            f"die gewählten Fenster ({wf_train_years} + {wf_test_years} Jahre)."
        )
    else:
        wf_metrics = compute_metrics(wf["returns"], wf["equity"])

        st.markdown("**Out-of-Sample-Kennzahlen (kombiniert)**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR (OOS)", fmt_pct(wf_metrics["cagr"]))
        c2.metric("Sharpe (OOS)", fmt_num(wf_metrics["sharpe"]))
        c3.metric("Max DD (OOS)", fmt_pct(wf_metrics["max_dd"]))
        c4.metric("Trades (OOS)", f"{int((wf['returns'] != 0).sum())}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wf["dates"], y=wf["equity"],
            name="Out-of-Sample-Equity", line=dict(color="#2ca02c", width=2),
        ))
        fig.update_layout(
            height=400, template="plotly_white", hovermode="x unified",
            yaxis_title="Kumulierter Wert",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Verwendete Parameter pro Fenster
        st.markdown("**Optimale Parameter pro Fenster**")
        param_df = pd.DataFrame([
            {"Fenster": i + 1, **p} for i, p in enumerate(wf["params"])
        ])
        st.dataframe(param_df, use_container_width=True, hide_index=True)

        st.info(
            "💡 **Interpretation:** Wenn die OOS-Performance deutlich schlechter ist als "
            "die In-Sample-Performance, ist die Strategie überangepasst. Eine robuste "
            "Strategie zeigt OOS-Ergebnisse ähnlich zur In-Sample-Phase."
        )


# ---------- TAB 4: MONTE CARLO ----------
with tab4:
    st.subheader("Monte-Carlo-Simulation")
    st.caption(
        "Block-Bootstrap der Tagesrenditen der Strategie. Zeigt die Verteilung "
        "möglicher Ergebnisse, wenn die Reihenfolge der Renditen zufällig variiert."
    )

    c1, c2 = st.columns(2)
    block_size = c1.slider("Blockgröße (Tage):", 5, 60, 20)
    n_sim = c2.slider("Anzahl Simulationen:", 500, 5000, 2000, step=500)

    with st.spinner("Simuliere..."):
        mc = monte_carlo_block_bootstrap(
            bt["returns"], block_size=block_size, n_sim=n_sim
        )

    if mc is None:
        st.warning("Zu wenig Daten für Monte-Carlo.")
    else:
        initial = bt["equity"].iloc[0]
        actual_final = bt["equity"].iloc[-1] / initial

        cagr_sims = mc["final"] ** (252 / len(bt["returns"])) - 1

        st.markdown("**Verteilung des Endwerts**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Median CAGR", fmt_pct(cagr_sims.median()))
        col2.metric("5%-Quantil CAGR", fmt_pct(cagr_sims.quantile(0.05)))
        col3.metric("95%-Quantil CAGR", fmt_pct(cagr_sims.quantile(0.95)))
        col4.metric("P(Verlust)", fmt_pct((mc["final"] < 1).mean()))

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=cagr_sims, nbinsx=60,
            marker_color="#1f77b4", opacity=0.75, name="Simulationen",
        ))
        fig.add_vline(
            x=metrics_strat["cagr"], line_dash="dash", line_color="red",
            annotation_text=f"Historisch: {metrics_strat['cagr']*100:.1f}%",
        )
        fig.update_layout(
            height=400, template="plotly_white",
            xaxis_title="CAGR", yaxis_title="Häufigkeit",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Verteilung des Max Drawdowns**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Median Max DD", fmt_pct(mc["max_dd"].median()))
        col2.metric("5%-Quantil Max DD", fmt_pct(mc["max_dd"].quantile(0.05)))
        col3.metric("Worst Case (Sim.)", fmt_pct(mc["max_dd"].min()))


# ---------- TAB 5: METHODIK & GRENZEN ----------
with tab5:
    st.markdown("""
### Wie dieser Backtest rechnet

1. **Signal-Generierung** auf Basis des Spreads = `INDEX_PC − EQUITY_PC`.
   Das Signal wird **um einen Tag verschoben**, bevor es auf die Renditen angewendet wird.
   Damit wird ausgeschlossen, dass der Backtest von Informationen profitiert, die zum
   Zeitpunkt der Entscheidung noch nicht bekannt waren (CBOE-Ratios werden erst nach
   Handelsschluss veröffentlicht).

2. **Handelbares Instrument:** SPY (mit Dividenden, `auto_adjust=True`).
   Nicht der Index selbst, denn `^GSPC` ist nicht direkt handelbar.

3. **Transaktionskosten:** Werden bei jeder Positionsänderung abgezogen, konfigurierbar
   über die Sidebar. Default 5 bp pro Trade.

4. **Kennzahlen:** CAGR = `(Endwert/Startwert)^(252/n) - 1`, Sharpe = `mean/std * sqrt(252)`,
   Max Drawdown = größter Peak-to-Trough-Verlust, Calmar = CAGR / |Max DD|.

5. **Walk-Forward:** Auf rollierenden Fenstern wird der beste Parameter bestimmt und
   auf das folgende, ungesehene Fenster angewendet.

6. **Monte Carlo:** Block-Bootstrap – es werden zufällige Blöcke aufeinanderfolgender
   Tagesrenditen gezogen, um die zeitliche Autokorrelation grob zu erhalten.

---

### Grenzen der Aussagekraft

- **Kurze Historie:** Equibles liefert maximal ~10 Jahre. Das entspricht einem einzigen
  Bullenmarkt mit einer kurzen Korrektur (2022). Die Strategie wurde **nicht** in einem
  langen Bärenmarkt wie 2000–2003 oder 2008 getestet.

- **Regime-Abhängigkeit:** Eine Strategie, die in einem Niedrigzins-Bullenmarkt gut
  funktioniert, kann in einem Hochzins-Bärenmarkt versagen.

- **Datenqualität:** Die Qualität der CBOE-Ratios via Equibles ist nicht unabhängig
  verifiziert. Kleine Fehler in den Rohdaten können den Backtest verzerren.

- **Kein Slippage-Modell:** Es wird ein fixer bp-Wert angenommen. In Stressphasen
  (Crashs) können die realen Kosten deutlich höher sein.

- **Kein Steuer-Effekt:** Kurzfristige Trades haben in Deutschland andere Steuerwirkungen
  als langfristiges Halten. Der Backtest ignoriert das.

- **Survivorship- und Selektions-Bias:** Nicht relevant hier, weil SPY und die
  CBOE-Ratios keine Auswahl unter vielen Alternativen sind.

- **Overfitting-Risiko:** Auch der Walk-Forward-Test kann Overfitting nicht vollständig
  ausschließen. Wenn die Strategie über viele Parameter optimiert wird, bleibt ein
  Restrisiko.

- **Monte Carlo ist kein Orakel:** Es sagt nur, wie sich die Ergebnisse verteilen
  würden, **wenn** die zukünftigen Renditen statistisch wie die historischen aussehen.
  Ein Regime-Wechsel wird nicht erfasst.

---

### Was dieses Skript **nicht** ist

Es ist **kein Beweis**, dass die Strategie in Zukunft funktioniert. Es ist ein
**Filter**, um zu sehen, ob die historische Evidenz überhaupt eine Spur von Edge
zeigt – und selbst das nur mit den oben genannten Einschränkungen. Wer eine
Strategie live einsetzt, sollte deutlich mehr Aufwand in Robustheit, Kostenmodellierung
und Risikomanagement stecken als in die Optimierung der Rendite.
    """)
