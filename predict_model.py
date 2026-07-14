#!/usr/bin/env python3
# ============================================================
# MODELLO PREDITTIVO — XGBoost su ticker selezionati (rendimento 30gg borsa)
# Lanciato da GitHub Actions con input "tickers" (separati da virgola) e
# "include_macro" (true/false). Scrive predictions.json alla radice.
#
# NOTA SUL MACRO: aggiunge 6 feature (VIX, tassi 10Y, dollaro, S&P) alle
# 10 tecniche gia' esistenti. Con un test set che ha gia' poche
# osservazioni indipendenti (~8, vedi INDEPENDENCE_WARNING), piu' feature
# = piu' rischio di overfitting, non meno. Per questo il flag e'
# confrontabile: lancia il modello con e senza macro e guarda se
# migliora DAVVERO o se il modello sta solo trovando pattern spuri.
# ============================================================
import sys
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

HORIZON = 30  # giorni di borsa, non calendario
OUT_FILE = Path(__file__).resolve().parent / "predictions.json"

TECH_FEATURES = ['EMA_10', 'EMA_50', 'RSI_14', 'VOL_10', 'VOL_21',
                  'VOL_CHANGE', 'RET_LAG_1', 'RET_LAG_2', 'RET_LAG_5', 'MOM_21']

# Serie macro condivise, scaricate UNA volta sola (non per ticker).
# ^TNX su Yahoo e' quotato a 10x il rendimento reale (es. 45.0 = 4.50%) --
# non importa per il modello (gli alberi sono invarianti alla scala),
# ma se guardi il numero grezzo non stupirti se sembra "alto".
MACRO_TICKERS = {"^VIX": "VIX", "^TNX": "TNX", "DX-Y.NYB": "DXY", "SPY": "SPY_MKT"}
MACRO_FEATURES = ['VIX_LEVEL', 'VIX_CHANGE_5D', 'TNX_LEVEL', 'TNX_CHANGE_21D',
                  'DXY_MOM_21', 'SPY_MOM_21']


def log(*a):
    print(*a, flush=True)


def build_features(data):
    log_ret = np.log(data['Close'] / data['Close'].shift(1))
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    data['RSI_14'] = 100 - (100 / (1 + avg_gain / avg_loss))

    data['VOL_10'] = log_ret.rolling(10).std()
    data['VOL_21'] = log_ret.rolling(21).std()
    data['VOL_CHANGE'] = data['Volume'].pct_change()
    data['RET_LAG_1'] = data['Close'].pct_change(1)
    data['RET_LAG_2'] = data['Close'].pct_change(2)
    data['RET_LAG_5'] = data['Close'].pct_change(5)
    data['MOM_21'] = data['Close'].pct_change(21)
    return data


def fetch_macro():
    """Scarica le serie macro condivise una sola volta per tutti i ticker."""
    try:
        raw = yf.download(list(MACRO_TICKERS.keys()), period="5y", progress=False, auto_adjust=True)["Close"]
    except Exception as e:
        log(f"  macro: download fallito ({e}), procedo senza.")
        return None
    if raw is None or raw.empty:
        return None
    macro = pd.DataFrame(index=raw.index)
    if "^VIX" in raw.columns:
        macro['VIX_LEVEL'] = raw["^VIX"]
        macro['VIX_CHANGE_5D'] = raw["^VIX"].pct_change(5)
    if "^TNX" in raw.columns:
        macro['TNX_LEVEL'] = raw["^TNX"]
        macro['TNX_CHANGE_21D'] = raw["^TNX"].diff(21)   # diff, non pct: e' gia' un tasso
    if "DX-Y.NYB" in raw.columns:
        macro['DXY_MOM_21'] = raw["DX-Y.NYB"].pct_change(21)
    if "SPY" in raw.columns:
        macro['SPY_MOM_21'] = raw["SPY"].pct_change(21)
    return macro


def run_for_ticker(ticker, macro_df, use_macro):
    log(f"\n--- {ticker} ---")
    raw = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
    if raw is None or raw.empty or len(raw) < 150:
        log(f"  dati insufficienti per {ticker}, salto.")
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    data = build_features(raw.copy())

    features = list(TECH_FEATURES)
    if use_macro and macro_df is not None:
        data = data.join(macro_df, how='left')
        features = features + [c for c in MACRO_FEATURES if c in data.columns]

    data['TARGET'] = data['Close'].shift(-HORIZON) / data['Close'] - 1
    model_df = data[features + ['TARGET']].dropna()

    if len(model_df) < 100:
        log(f"  righe utilizzabili insufficienti ({len(model_df)}) per {ticker}, salto.")
        return None

    split_idx = int(len(model_df) * 0.8)
    train = model_df.iloc[: split_idx - HORIZON]
    test = model_df.iloc[split_idx:]
    if len(train) < 50 or len(test) < 10:
        log(f"  train/test troppo piccoli per {ticker}, salto.")
        return None

    X_train, y_train = train[features], train['TARGET']
    X_test, y_test = test[features], test['TARGET']

    model = XGBRegressor(max_depth=3, learning_rate=0.05, n_estimators=100,
                          subsample=0.8, colsample_bytree=0.8,
                          objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_r2 = float(r2_score(y_test, baseline_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    eff_n = len(test) / HORIZON

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    series = [{"date": d.strftime("%Y-%m-%d"), "actual": float(a), "predicted": float(p)}
              for d, a, p in zip(test.index, y_test.values, y_pred)]

    log(f"  R2={r2:.4f} (baseline {baseline_r2:.4f}) | RMSE={rmse:.4f} | eff.N~{eff_n:.1f} | feature={len(features)}")
    if len(features) > eff_n:
        log(f"  >> ATTENZIONE: {len(features)} feature ma solo ~{eff_n:.1f} osservazioni indipendenti nel test.")
        log("     Overfitting altamente probabile: qualunque R2 va preso con molta cautela.")

    return {
        "ticker": ticker,
        "horizon": HORIZON,
        "include_macro": bool(use_macro),
        "n_features": len(features),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "effective_n_test": round(eff_n, 1),
        "r2": round(r2, 4),
        "rmse": round(rmse, 5),
        "baseline_r2": round(baseline_r2, 4),
        "baseline_rmse": round(baseline_rmse, 5),
        "beats_baseline": bool(r2 > baseline_r2),
        "feature_importances": {k: round(float(v), 4) for k, v in importances.items()},
        "series": series,
    }


def main():
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        raise SystemExit("Uso: python predict_model.py TICKER1,TICKER2,... [true|false]")
    tickers = [t.strip().upper() for t in sys.argv[1].split(",") if t.strip()]
    use_macro = True
    if len(sys.argv) >= 3:
        use_macro = sys.argv[2].strip().lower() in ("true", "1", "yes", "si", "sì")
    log(f"Ticker richiesti: {tickers} | include_macro={use_macro}")

    macro_df = fetch_macro() if use_macro else None
    if use_macro and macro_df is None:
        log("  macro richiesto ma non disponibile: procedo solo con feature tecniche.")

    results = []
    for t in tickers:
        try:
            r = run_for_ticker(t, macro_df, use_macro)
            if r:
                results.append(r)
        except Exception as e:
            log(f"  ERRORE su {t}: {e}")

    out = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "horizon_trading_days": HORIZON,
        "include_macro": use_macro,
        "requested_tickers": tickers,
        "results": results,
    }
    OUT_FILE.write_text(json.dumps(out, indent=2))
    log(f"\nScritto {OUT_FILE.name} con {len(results)}/{len(tickers)} ticker completati.")


if __name__ == "__main__":
    main()
