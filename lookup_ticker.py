#!/usr/bin/env python3
# ============================================================
# LOOKUP — punteggio on-demand per uno o più ticker, separati da virgola
# (stesso formato di predict.yml). Lanciato da GitHub Actions con input
# "ticker" (workflow_dispatch). Scrive lookup.json alla radice.
#
# NON rifà il fetch dell'intero universo Russell 3000 (10+ minuti):
# scarica SOLO i ticker richiesti da EDGAR + Yahoo Finance (pochi secondi
# l'uno), poi li z-scora contro i peer del loro settore già presenti
# nell'ultimo ranking.csv committato -- stessa identica metodologia di
# build_ranking.py (stesso cap dei ratio, stesso quantile clip 2%/98%,
# stessa soglia cnt>=6 per il fallback a statistiche globali). Il
# risultato e' "che punteggio avrebbe avuto se fosse stato incluso
# nell'ultimo run settimanale", non un nuovo run completo. Un ticker che
# fallisce (CIK non trovato, EDGAR senza dati us-gaap, ecc.) non blocca
# gli altri: finisce in "errors", il resto prosegue.
#
# LIMITI DA SAPERE:
# - Se un ticker non e' gia' nell'universo Russell 3000, non ho un
#   settore GICS-like affidabile da EDGAR/Yahoo -- lo score usa le
#   statistiche globali (tutti i settori insieme) invece di quelle del
#   suo settore, marcato esplicitamente in output.
# - ranking.csv puo' avere fino a una settimana di ritardo: i peer con
#   cui confronti il ticker sono quelli dell'ultimo lunedi', non di oggi.
# - Nessun trap_flag: servirebbe il fscore_n>=6 E la soglia dei percentili
#   dell'intero universo ricalcolata -- qui do fscore/fscore_n grezzi,
#   la lettura "trappola o occasione" resta un giudizio da fare a mano.
# ============================================================
import sys
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from build_ranking import (
    facts_of, ttm, instant, compute_fscore,
    REV, NI, OPI, GP, CFO, CAPEX, EQ, ASSETS, DEBT, CASH, TAX, PRETAX,
    SEC_TICKERS_URL, UA, WEIGHTS, MISSING_PENALTY,
)

OUT_DIR = Path(__file__).resolve().parent
RANKING_FILE = OUT_DIR / "ranking.csv"

# stessi cap "di sanita'" applicati in build_ranking.py prima dello z-score
CAPS = {
    "earn_yield": (-0.25, 0.25), "fcf_yield": (-0.25, 0.25), "ebit_yield": (-0.30, 0.30),
    "book_price": (0, 3), "roic": (-0.5, 0.60), "roe": (-1.0, 0.80),
    "op_margin": (-1.0, 1.0), "gross_margin": (-1.0, 1.0), "debt_eq": (0, 5),
}


def log(*a):
    print(*a, flush=True)


def peer_stats(ranking, sector, col):
    """Replica zscore_col di build_ranking.py, ma sui soli peer di
    settore gia' presenti in ranking.csv: stesso quantile clip 2%/98%,
    stessa soglia cnt>=6 per il fallback a statistiche globali."""
    x_all = pd.to_numeric(ranking[col], errors="coerce") if col in ranking.columns else pd.Series(dtype=float)
    lo_g, hi_g = (x_all.quantile(0.02), x_all.quantile(0.98)) if len(x_all.dropna()) else (np.nan, np.nan)
    x_all_c = x_all.clip(lo_g, hi_g)
    peers = ranking[ranking["sector"] == sector] if sector is not None else ranking.iloc[0:0]
    x_sec = pd.to_numeric(peers[col], errors="coerce") if col in peers.columns else pd.Series(dtype=float)
    n_sec = int(x_sec.notna().sum())
    if n_sec >= 6:
        lo, hi = x_sec.quantile(0.02), x_sec.quantile(0.98)
        x_sec_c = x_sec.clip(lo, hi)
        return lo, hi, x_sec_c.mean(), x_sec_c.std(), n_sec, "settore"
    return lo_g, hi_g, x_all_c.mean(), x_all_c.std(ddof=0), n_sec, "globale (fallback, <6 peer nel settore)"


def zscore_one(raw_value, ranking, sector, col, sign=1):
    if raw_value is None or pd.isna(raw_value):
        return MISSING_PENALTY, "n/d"
    lo, hi, mean, std, n_sec, basis = peer_stats(ranking, sector, col)
    if pd.isna(lo) or pd.isna(mean) or pd.isna(std) or std == 0:
        return 0.0, basis
    clipped = min(max(raw_value, lo), hi)
    z = (clipped - mean) / std
    return float(np.clip(z * sign, -3, 3)), basis


def lookup_one(ticker, ranking, ticker_to_cik):
    cik = ticker_to_cik.get(ticker)
    if not cik:
        raise ValueError(f"Nessun CIK SEC trovato per {ticker}. Controlla lo ticker -- alcuni "
                          f"titoli multi-classe usano simboli diversi tra fonti (es. BRKB vs BRK-B).")

    log(f"  fondamentali EDGAR (CIK {cik})...")
    facts = facts_of(cik)
    gaap = facts.get("us-gaap", {})
    dei_facts = facts.get("dei", {})
    if not gaap:
        raise ValueError(f"EDGAR non ha dati us-gaap per {ticker} -- probabilmente non e' un "
                          f"filer SEC domestico (10-K/10-Q), es. un ADR che deposita 20-F.")

    rev, ni, opi, gp = ttm(gaap, REV), ttm(gaap, NI), ttm(gaap, OPI), ttm(gaap, GP)
    cfo, capex = ttm(gaap, CFO), ttm(gaap, CAPEX)
    tax, pretax = ttm(gaap, TAX), ttm(gaap, PRETAX)
    eq, assets = instant(gaap, EQ), instant(gaap, ASSETS)
    debt = instant(gaap, DEBT) or 0
    cash = instant(gaap, CASH) or 0
    shares = instant(dei_facts, ["EntityCommonStockSharesOutstanding"]) or instant(
        gaap, ["CommonStockSharesOutstanding"])
    fcf = (cfo - capex) if (cfo is not None and capex is not None) else None
    taxrate = (tax / pretax) if (tax is not None and pretax not in (None, 0)) else 0.21
    taxrate = min(max(taxrate, 0), 0.5)
    nopat = opi * (1 - taxrate) if opi is not None else None
    invested = (eq + debt - cash) if eq is not None else None
    fscore, fscore_n = compute_fscore(gaap)

    log("  prezzo/momentum da Yahoo Finance...")
    yf_ticker = ticker.replace(".", "-")
    px = yf.download(yf_ticker, period="400d", progress=False, auto_adjust=True)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px = px.dropna()
    price = float(px.iloc[-1]) if len(px) else None
    mom = float(px.iloc[-21] / px.iloc[-252] - 1) if len(px) >= 260 else None
    window = px.iloc[-252:] if len(px) >= 252 else px
    high_52w = float(window.max()) if len(window) else None
    low_52w = float(window.min()) if len(window) else None

    if price is None or not shares:
        raise ValueError(f"Prezzo o azioni in circolazione mancanti per {ticker} -- impossibile "
                          f"calcolare la market cap, quindi nessuno score.")

    mktcap = price * shares
    ev = mktcap + debt - cash
    pos_eq = eq is not None and eq > 0

    raw = {
        "earn_yield": (ni / mktcap) if ni is not None else None,
        "fcf_yield": (fcf / mktcap) if fcf is not None else None,
        "ebit_yield": (opi / ev) if (opi is not None and ev) else None,
        "book_price": (eq / mktcap) if pos_eq else None,
        "roe": (ni / eq) if (pos_eq and ni is not None) else None,
        "roic": (nopat / invested) if (nopat is not None and invested and invested > 0) else None,
        "gross_margin": (gp / rev) if (gp is not None and rev) else None,
        "op_margin": (opi / rev) if (opi is not None and rev) else None,
        "debt_eq": (debt / eq) if pos_eq else None,
        "mom": mom,
    }
    for k, (lo, hi) in CAPS.items():
        if raw.get(k) is not None:
            raw[k] = float(np.clip(raw[k], lo, hi))

    # settore: se il ticker e' gia' nell'universo Russell 3000 uso quello
    # (coerente col resto del batch); altrimenti nessuna fonte GICS-like
    # affidabile qui -- fallback esplicito alle statistiche globali.
    if ticker in ranking.index:
        sector = str(ranking.loc[ticker, "sector"])
        in_universe = True
    else:
        sector = None
        in_universe = False
        log(f"  >> ATTENZIONE: {ticker} non e' nell'universo Russell 3000 corrente -- "
            f"z-score calcolato contro le statistiche GLOBALI (tutti i settori), non contro "
            f"i peer di settore. Meno preciso, ma esplicito.")

    value_parts, quality_parts = [], []
    bases = {}
    for col in ["earn_yield", "fcf_yield", "ebit_yield", "book_price"]:
        z, basis = zscore_one(raw.get(col), ranking, sector, col)
        value_parts.append(z)
        bases[col] = basis
    for col, sign in [("roic", 1), ("roe", 1), ("op_margin", 1), ("gross_margin", 1), ("debt_eq", -1)]:
        z, basis = zscore_one(raw.get(col), ranking, sector, col, sign=sign)
        quality_parts.append(z)
        bases[col] = basis
    momentum_score, mom_basis = zscore_one(raw.get("mom"), ranking, sector, "mom")
    bases["mom"] = mom_basis

    value_score = float(np.mean(value_parts))
    quality_score = float(np.mean(quality_parts))
    score = WEIGHTS["value"] * value_score + WEIGHTS["quality"] * quality_score + WEIGHTS["momentum"] * momentum_score

    rank_if_inserted = int((ranking["SCORE"] > score).sum()) + 1
    n_total = len(ranking) + 1

    return {
        "ticker": ticker,
        "in_universe_russell3000": in_universe,
        "sector": sector or "sconosciuto (fallback globale)",
        "z_score_basis": bases,
        "raw_metrics": {k: (round(v, 4) if v is not None else None) for k, v in raw.items()},
        "value_score": round(value_score, 4),
        "quality_score": round(quality_score, 4),
        "momentum_score": round(momentum_score, 4),
        "SCORE": round(score, 4),
        "rank_if_inserted": rank_if_inserted,
        "n_total_if_inserted": n_total,
        "fscore": fscore,
        "fscore_n": fscore_n,
        "price": price, "high_52w": high_52w, "low_52w": low_52w, "mktcap": round(mktcap, 0),
        "note": "Score calcolato contro i peer di settore gia' presenti nell'ultimo ranking.csv "
                "committato (fino a una settimana di ritardo) -- non un nuovo run completo. "
                "Nessun trap_flag: leggi fscore/fscore_n a mano insieme al value_score.",
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python lookup_ticker.py TICKER1,TICKER2,...")
    requested = [t.strip().upper() for t in sys.argv[1].split(",") if t.strip()]
    if not requested:
        raise SystemExit("Nessun ticker valido nell'input.")

    if not RANKING_FILE.exists():
        raise SystemExit("ranking.csv non trovato -- lancia prima il workflow 'Aggiorna ranking'.")
    ranking = pd.read_csv(RANKING_FILE, index_col=0)

    log("Scarico la mappa ticker -> CIK da SEC (una volta sola per tutti i ticker richiesti)...")
    sec_map = requests.get(SEC_TICKERS_URL, headers=UA, timeout=60).json()
    ticker_to_cik = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in sec_map.values()}

    results, errors = [], []
    for ticker in requested:
        log(f"\n--- {ticker} ---")
        try:
            r = lookup_one(ticker, ranking, ticker_to_cik)
            results.append(r)
            log(f"  SCORE={r['SCORE']:.4f}  (value {r['value_score']:+.2f} / quality {r['quality_score']:+.2f} "
                f"/ momentum {r['momentum_score']:+.2f})  -- rank_if_inserted {r['rank_if_inserted']}/{r['n_total_if_inserted']}")
        except Exception as e:
            log(f"  >> ERRORE: {e}")
            errors.append({"ticker": ticker, "error": str(e)})

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "requested_tickers": requested,
        "results": results,
        "errors": errors,
    }
    with open(OUT_DIR / "lookup.json", "w") as f:
        json.dump(payload, f, indent=2)

    log(f"\nFatto: {len(results)}/{len(requested)} ticker trovati. Scritto lookup.json.")
    if errors:
        log("Ticker con errore: " + ", ".join(e["ticker"] for e in errors))


if __name__ == "__main__":
    main()
