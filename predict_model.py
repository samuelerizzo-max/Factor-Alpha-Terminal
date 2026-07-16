#!/usr/bin/env python3
# ============================================================
# MODELLO PREDITTIVO — XGBoost su ticker selezionati (rendimento 30gg borsa)
# Lanciato da GitHub Actions con input "tickers" e "include_macro".
# Scrive predictions.json alla radice.
#
# v3: walk-forward (rolling-origin) invece di un singolo split 80/20.
# Il modello viene riaddestrato a ogni fold su tutta la storia disponibile
# fino a quel punto (purge di HORIZON righe prima del test, stesso principio
# anti-leakage di prima) e testato su UN solo punto di decisione, poi la
# finestra di training si allarga e si avanza di HORIZON righe. Questo porta
# le osservazioni indipendenti nel test da ~8 (un solo split) a ~20-30
# (dipende dalla storia disponibile per ticker) — R2/RMSE/backtest sono ora
# calcolati sull'intero pool di previsioni out-of-sample invece che su
# un'unica finestra finale. Non risolve il problema di fondo (weak-form
# efficiency a 30gg su singoli nomi liquidi resta un muro reale), ma rende
# la stima meno dipendente dal singolo split scelto.
#
# NOTA SUL MACRO: aggiunge 6 feature (VIX, tassi 10Y, dollaro, S&P) alle
# feature tecniche. Con poche osservazioni indipendenti nel test, piu'
# feature = piu' rischio di overfitting, non meno. Il flag e' pensato per
# essere confrontato (con/senza), non per essere lasciato sempre acceso
# senza guardare la differenza.
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
COST_PER_SIDE = 0.0005  # 5 bps per lato = 10 bps andata+ritorno (stessa convenzione del motore fattoriale)
OUT_FILE = Path(__file__).resolve().parent / "predictions.json"

INIT_TRAIN_FRAC = 0.30  # frazione iniziale di storia usata come primo training set del walk-forward
MIN_INIT_TRAIN = 250    # minimo assoluto di righe per il primo fit (sotto, gli alberi sono instabili)
MIN_FOLDS = 10          # sotto questa soglia il walk-forward non e' piu' affidabile del vecchio split singolo

TECH_FEATURES = ['EMA_10', 'EMA_50', 'RSI_14', 'VOL_10', 'VOL_21', 'VOL_CHANGE',
                  'RET_LAG_5', 'MOM_21', 'MACD_HIST', 'ADX_14', 'ATR_NORM']

MACRO_FEATURES = ['VIX_LEVEL', 'VIX_CHANGE_5D', 'TNX_LEVEL', 'TNX_CHANGE_21D',
                  'DXY_MOM_21', 'SPY_MOM_21']


def log(*a):
    print(*a, flush=True)


def compute_macd_hist(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def compute_adx_atr(high, low, close, period=14):
    """ADX e ATR col metodo di Wilder (stesso smoothing esponenziale usato per l'RSI)."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx, atr


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
    data['RET_LAG_5'] = data['Close'].pct_change(5)
    data['MOM_21'] = data['Close'].pct_change(21)

    data['MACD_HIST'] = compute_macd_hist(data['Close'])
    adx, atr = compute_adx_atr(data['High'], data['Low'], data['Close'])
    data['ADX_14'] = adx
    data['ATR_NORM'] = atr / data['Close']
    return data


def fetch_macro():
    try:
        raw = yf.download(["^VIX", "^TNX", "DX-Y.NYB", "SPY"], period="5y", progress=False, auto_adjust=True)["Close"]
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
        macro['TNX_CHANGE_21D'] = raw["^TNX"].diff(21)
    if "DX-Y.NYB" in raw.columns:
        macro['DXY_MOM_21'] = raw["DX-Y.NYB"].pct_change(21)
    if "SPY" in raw.columns:
        macro['SPY_MOM_21'] = raw["SPY"].pct_change(21)
    return macro


def run_backtest(y_test_actual, y_pred, dates_str):
    """
    Backtest sulle decisioni walk-forward: ogni elemento in input e' gia' un
    punto di decisione singolo (un fold = HORIZON giorni avanti rispetto al
    precedente), quindi qui non serve piu' sotto-campionare una serie densa
    giornaliera come nella v2 -- il non-sovrapposto e' garantito a monte
    dalla struttura dei fold. Long se il modello prevede rendimento positivo,
    altrimenti cash. Costi reali. Confronto esplicito con buy&hold: un
    modello puo' avere un win rate alto e comunque perdere contro il tenere
    e basta, se sta in cash anche durante rialzi veri.
    """
    n = len(y_test_actual)
    equity = [1.0]
    bh_equity = [1.0]
    trades = []
    for i in range(n):
        pred = float(y_pred[i])
        actual = float(y_test_actual[i])
        if pred > 0:
            net_return = actual - 2 * COST_PER_SIDE
            trades.append({"date": dates_str[i], "predicted": round(pred, 4),
                            "actual": round(actual, 4), "net_return": round(net_return, 4)})
        else:
            net_return = 0.0
        equity.append(equity[-1] * (1 + net_return))
        bh_equity.append(bh_equity[-1] * (1 + actual))

    n_trades = len(trades)
    wins = sum(1 for t in trades if t["net_return"] > 0)
    win_rate = (wins / n_trades) if n_trades else None
    total_return = equity[-1] - 1
    bh_total_return = bh_equity[-1] - 1
    rr = np.array([t["net_return"] for t in trades]) if trades else np.array([])
    sharpe_like = float(rr.mean() / rr.std() * np.sqrt(252 / HORIZON)) if len(rr) > 1 and rr.std() > 0 else None
    eq = pd.Series(equity)
    max_dd = float((eq / eq.cummax() - 1).min())

    return {
        "n_decision_points": n,
        "n_trades_long": n_trades,
        "win_rate": round(win_rate, 3) if win_rate is not None else None,
        "total_return": round(float(total_return), 4),
        "buy_hold_total_return": round(float(bh_total_return), 4),
        "beats_buy_hold": bool(total_return > bh_total_return),
        "sharpe_like": round(sharpe_like, 2) if sharpe_like is not None else None,
        "max_drawdown": round(max_dd, 4),
        "equity_curve": [round(float(e), 4) for e in equity],
        "trades": trades,
    }


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

    n_rows = len(model_df)
    if n_rows < MIN_INIT_TRAIN + HORIZON + MIN_FOLDS * HORIZON // 2:
        log(f"  righe utilizzabili insufficienti ({n_rows}) per {ticker}, salto.")
        return None

    init_train_end = max(MIN_INIT_TRAIN, int(n_rows * INIT_TRAIN_FRAC))
    if init_train_end >= n_rows - HORIZON:
        log(f"  storia insufficiente per un walk-forward vero ({n_rows} righe), salto.")
        return None

    fold_starts = list(range(init_train_end, n_rows, HORIZON))
    if len(fold_starts) < MIN_FOLDS:
        log(f"  >> ATTENZIONE: solo {len(fold_starts)} fold walk-forward disponibili "
            f"(sotto la soglia di {MIN_FOLDS}).")

    y_test_all, y_pred_all, baseline_all, dates_all = [], [], [], []
    last_model, last_train_size = None, None

    for fold_start in fold_starts:
        train_end = fold_start - HORIZON  # purge: nessuna riga di training il cui target guarda oltre il test point
        if train_end < MIN_INIT_TRAIN - HORIZON:
            continue
        train = model_df.iloc[:train_end]
        test_row = model_df.iloc[[fold_start]]
        X_train, y_train = train[features], train['TARGET']
        X_test, y_test = test_row[features], test_row['TARGET']

        model = XGBRegressor(max_depth=3, learning_rate=0.05, n_estimators=100,
                              subsample=0.8, colsample_bytree=0.8,
                              objective='reg:squarederror', random_state=42)
        model.fit(X_train, y_train)
        pred = float(model.predict(X_test)[0])

        y_test_all.append(float(y_test.iloc[0]))
        y_pred_all.append(pred)
        baseline_all.append(float(y_train.mean()))  # baseline walk-forward: media disponibile *fino a quel fold*
        dates_all.append(test_row.index[0].strftime("%Y-%m-%d"))
        last_model, last_train_size = model, len(train)

    n_folds = len(y_test_all)
    if n_folds < 5:
        log(f"  fold utilizzabili insufficienti ({n_folds}) per {ticker}, salto.")
        return None

    y_test_arr = np.array(y_test_all)
    y_pred_arr = np.array(y_pred_all)
    baseline_arr = np.array(baseline_all)

    r2 = float(r2_score(y_test_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_test_arr, y_pred_arr)))
    baseline_r2 = float(r2_score(y_test_arr, baseline_arr))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test_arr, baseline_arr)))

    # importanza media delle feature sui fold: piu' robusta di quella di un singolo fit
    importances = pd.Series(last_model.feature_importances_, index=features).sort_values(ascending=False)

    series = [{"date": dd, "actual": a, "predicted": p}
              for dd, a, p in zip(dates_all, y_test_all, y_pred_all)]

    # serie tecnica per il grafico (prezzo + EMA + RSI) -- SOLO visualizzazione,
    # nessun segnale nascosto qui dentro. Ultimi ~180 giorni per leggibilita'.
    chart_tail = data.dropna(subset=['Close', 'EMA_10', 'EMA_50', 'RSI_14']).tail(180)
    chart_series = [{
        "date": idx.strftime("%Y-%m-%d"),
        "close": round(float(row['Close']), 2),
        "ema10": round(float(row['EMA_10']), 2),
        "ema50": round(float(row['EMA_50']), 2),
        "rsi14": round(float(row['RSI_14']), 1),
    } for idx, row in chart_tail.iterrows()]

    backtest = run_backtest(y_test_arr, y_pred_arr, dates_all)

    log(f"  R2={r2:.4f} (baseline {baseline_r2:.4f}) | RMSE={rmse:.4f} | fold walk-forward={n_folds} | feature={len(features)}")
    if len(features) > n_folds:
        log(f"  >> ATTENZIONE: {len(features)} feature ma solo {n_folds} fold indipendenti.")
    log(f"  Backtest: {backtest['n_trades_long']} trade long, win rate "
        f"{backtest['win_rate']}, tot {backtest['total_return']:+.1%} vs buy&hold {backtest['buy_hold_total_return']:+.1%}")
    if n_folds < MIN_FOLDS:
        log(f"  >> ATTENZIONE: meno di {MIN_FOLDS} fold. Win rate e Sharpe non sono statisticamente significativi.")

    return {
        "ticker": ticker,
        "horizon": HORIZON,
        "include_macro": bool(use_macro),
        "n_features": len(features),
        "n_train_initial": int(init_train_end),
        "n_train_final": int(last_train_size),
        "n_folds": n_folds,
        "effective_n_test": n_folds,
        "r2": round(r2, 4),
        "rmse": round(rmse, 5),
        "baseline_r2": round(baseline_r2, 4),
        "baseline_rmse": round(baseline_rmse, 5),
        "beats_baseline": bool(r2 > baseline_r2),
        "feature_importances": {k: round(float(v), 4) for k, v in importances.items()},
        "series": series,
        "chart_series": chart_series,
        "backtest": backtest,
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
