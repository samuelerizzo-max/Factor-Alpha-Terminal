#!/usr/bin/env python3
# ============================================================
# FACTOR ALPHA — INTERNAZIONALE (Europa, EM incl. EM-Asia, GCC ex-UAE,
# Giappone, Canada, Australia) — pipeline MANUALE, non su GitHub Actions.
#
# Prende in input un export Bloomberg EQS (stesso formato gia' usato nelle
# sessioni di screening precedenti: GICS_SECTOR_NAME, CNTRY_OF_DOMICILE,
# PX_TO_BOOK_RATIO, BEST_EV_TO_BEST_EBITDA, RETURN_COM_EQY,
# NET_DEBT_TO_EBITDA, GROSS_MARGIN, FREE_CASH_FLOW_YIELD, SALES_GROWTH,
# EQY_REC_CONS -- o gli equivalenti header in inglese semplice) e applica
# LA STESSA metodologia value 40 / quality 40 / momentum 20, z-score
# sector-neutral, penalita' sui fattori mancanti, di build_ranking.py.
#
# Output: un CSV con lo schema ESATTO che la tab SCREEN della dashboard
# gia' accetta col bottone "Importa ranking.csv" (nessuna modifica a
# index.html). Da caricare manualmente, quando serve -- niente cron.
#
# ATTENZIONE SULLE ETICHETTE: 'ebit_yield' e 'debt_eq' nella UI sono
# etichettati "EBIT/EV Yield" e "Debt / Equity". Qui non abbiamo EBIT/EV
# ne' Debt/Equity puro -- abbiamo EV/EBITDA e Net Debt/EBITDA. Li uso per
# calcolare value_score/quality_score (corretto), ma NON li scrivo in
# quelle due colonne con l'etichetta sbagliata: meglio ometterle che
# mostrarle mislabeled. earn_yield (P/E fwd) e roic non sono popolati per
# lo stesso motivo di coerenza (earn_yield qui sarebbe forward, non
# trailing come nel motore US; roic non calcolabile da EQS senza NOPAT).
# ============================================================
import sys
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

WEIGHTS = {"value": 0.40, "quality": 0.40, "momentum": 0.20}
MISSING_PENALTY = -0.75
OUT_DIR = Path(__file__).resolve().parent

# esclusione UAE per compliance (universo dichiarato: GCC ex-UAE)
UAE_TOKENS = {"united arab emirates", "uae", "u.a.e.", "ae"}
# nomi/paesi che richiedono controllo MNPI auto-amministrato (ruolo ECM)
ITALY_TOKENS = {"italy", "italia", "it"}


def log(*a):
    print(*a, flush=True)


ALIASES = {
    "ticker": ["Ticker", "ticker", "TICKER"],
    "name": ["Name", "Nome", "name", "Security Name"],
    "country": ["Country", "CNTRY_OF_DOMICILE", "country"],
    "sector": ["Sector", "GICS_SECTOR_NAME", "sector"],
    "mktcap": ["Market Cap", "MarketCap", "CUR_MKT_CAP", "market_cap"],
    "pe_fwd": ["P/E fwd", "PE fwd", "BEst P/E forward", "BEST_PE_RATIO", "PE_FWD"],
    "pb": ["P/B", "PX_TO_BOOK_RATIO", "PB"],
    "ev_ebitda": ["EV/EBITDA", "BEST_EV_TO_BEST_EBITDA", "EV_EBITDA"],
    "roe": ["ROE", "RETURN_COM_EQY"],
    "net_debt_ebitda": ["Net Debt/EBITDA", "NET_DEBT_TO_EBITDA", "NetDebt/EBITDA"],
    "gross_margin": ["Gross Margin", "GROSS_MARGIN"],
    "fcf_yield": ["FCF Yield", "Free Cash Flow Yield", "FREE_CASH_FLOW_YIELD"],
    "sales_growth": ["Sales Growth", "Revenue Growth YoY", "SALES_GROWTH"],
    "consensus": ["Analyst Consensus", "EQY_REC_CONS", "Consensus"],
    "price": ["Last Price", "PX_LAST", "Price"],
    "high_52w": ["52-week High", "52W High", "HIGH_52WEEK", "PX_HIGH_52WEEK"],
    "low_52w": ["52-week Low", "52W Low", "LOW_52WEEK", "PX_LOW_52WEEK"],
    "chg_1yr": ["1Y Price Change", "CHG_PCT_1YR", "1Y Change"],
}


def find_col(df, keys):
    lower_map = {c.lower().strip(): c for c in df.columns}
    for k in ALIASES[keys]:
        if k.lower().strip() in lower_map:
            return lower_map[k.lower().strip()]
    return None


def as_ratio(series):
    """Bloomberg esporta le percentuali come numeri interi (15 = 15%).
    Se la mediana assoluta e' > 1.5 assumo percentuale e divido per 100;
    altrimenti assumo che sia gia' in forma decimale."""
    s = pd.to_numeric(series, errors="coerce")
    med = s.abs().median(skipna=True)
    if pd.notna(med) and med > 1.5:
        return s / 100.0
    return s


def zscore_col(df, sec, cnt, col, sign=1):
    x = pd.to_numeric(df[col], errors="coerce")
    x = x.clip(x.quantile(0.02), x.quantile(0.98))
    g = x.groupby(sec)
    zsec = (x - g.transform("mean")) / g.transform("std")
    zglob = (x - x.mean()) / x.std(ddof=0)
    z = zsec.where(cnt >= 6, zglob)
    return (z * sign).clip(-3, 3).fillna(MISSING_PENALTY)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python build_ranking_intl.py export_bloomberg.xlsx [output.csv]")
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else OUT_DIR / "ranking_intl.csv"

    log(f"Leggo {in_path.name}...")
    raw = pd.read_excel(in_path) if in_path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(in_path)
    raw.columns = [str(c).strip() for c in raw.columns]
    log(f"  {len(raw)} righe, colonne: {list(raw.columns)}")

    cols = {k: find_col(raw, k) for k in ALIASES}
    missing_required = [k for k in ("ticker", "sector", "pe_fwd", "pb", "ev_ebitda", "roe") if cols[k] is None]
    if missing_required:
        raise SystemExit(f"Colonne mancanti indispensabili: {missing_required}. "
                          f"Controlla i nomi delle colonne nel file (vedi ALIASES nello script).")

    df = pd.DataFrame({"ticker": raw[cols["ticker"]].astype(str).str.strip().str.upper()})
    df["sector"] = raw[cols["sector"]].fillna("Unknown").astype(str).str.strip()
    df["country"] = raw[cols["country"]].astype(str).str.strip() if cols["country"] else "Unknown"

    # ---- esclusione UAE per compliance (universo GCC ex-UAE) ----
    n_before = len(df)
    is_uae = df["country"].str.lower().isin(UAE_TOKENS)
    df = df[~is_uae].copy()
    raw = raw.loc[df.index]
    if is_uae.sum():
        log(f"  esclusi {int(is_uae.sum())} nomi UAE per compliance (GCC ex-UAE).")
    log(f"  universo dopo esclusioni: {len(df)}/{n_before}")

    # ---- fattori grezzi ----
    pe_fwd = pd.to_numeric(raw[cols["pe_fwd"]], errors="coerce")
    pb = pd.to_numeric(raw[cols["pb"]], errors="coerce")
    ev_ebitda = pd.to_numeric(raw[cols["ev_ebitda"]], errors="coerce")
    df["earn_yield_fwd"] = np.where(pe_fwd > 0, 1.0 / pe_fwd, np.nan)      # forward, non trailing come nel motore US
    df["book_price"] = np.where(pb > 0, 1.0 / pb, np.nan)
    df["ebitda_ev_yield"] = np.where(ev_ebitda > 0, 1.0 / ev_ebitda, np.nan)  # EBITDA/EV, non EBIT/EV
    df["fcf_yield"] = as_ratio(raw[cols["fcf_yield"]]) if cols["fcf_yield"] else np.nan

    df["roe"] = as_ratio(raw[cols["roe"]])
    df["gross_margin"] = as_ratio(raw[cols["gross_margin"]]) if cols["gross_margin"] else np.nan
    df["net_debt_ebitda"] = pd.to_numeric(raw[cols["net_debt_ebitda"]], errors="coerce") if cols["net_debt_ebitda"] else np.nan

    df["mktcap"] = pd.to_numeric(raw[cols["mktcap"]], errors="coerce") if cols["mktcap"] else np.nan
    df["price"] = pd.to_numeric(raw[cols["price"]], errors="coerce") if cols["price"] else np.nan
    df["high_52w"] = pd.to_numeric(raw[cols["high_52w"]], errors="coerce") if cols["high_52w"] else np.nan
    df["low_52w"] = pd.to_numeric(raw[cols["low_52w"]], errors="coerce") if cols["low_52w"] else np.nan

    # momentum: preferisco il cambio 1Y diretto; altrimenti posizione nel range 52w
    # (proxy piu' debole -- niente esclusione dell'ultimo mese come nel 12-1 US,
    # e la distanza dai massimi/minimi resta un rischio di anchoring se presa da
    # sola: qui e' solo il 20% del composito, mai un segnale a se stante).
    if cols["chg_1yr"]:
        df["mom"] = as_ratio(raw[cols["chg_1yr"]])
        mom_source = "1Y price change diretto"
    elif cols["high_52w"] and cols["low_52w"] and cols["price"]:
        rng = df["high_52w"] - df["low_52w"]
        df["mom"] = np.where(rng > 0, (df["price"] - df["low_52w"]) / rng, np.nan)
        mom_source = "posizione nel range 52w (proxy, non 12-1)"
    else:
        df["mom"] = np.nan
        mom_source = "non disponibile (colonne mancanti)"
    log(f"  momentum: {mom_source}")

    df = df[(df["mktcap"].fillna(0) > 0) | df["mktcap"].isna()]  # non escludo se mktcap manca, solo se e' 0/negativo

    sec = df["sector"]
    cnt = sec.groupby(sec).transform("count")

    value = pd.concat([
        zscore_col(df, sec, cnt, "earn_yield_fwd"),
        zscore_col(df, sec, cnt, "fcf_yield"),
        zscore_col(df, sec, cnt, "ebitda_ev_yield"),
        zscore_col(df, sec, cnt, "book_price"),
    ], axis=1).mean(axis=1)

    quality_parts = [zscore_col(df, sec, cnt, "roe"), zscore_col(df, sec, cnt, "gross_margin")]
    if df["net_debt_ebitda"].notna().any():
        quality_parts.append(zscore_col(df, sec, cnt, "net_debt_ebitda", sign=-1))
    quality = pd.concat(quality_parts, axis=1).mean(axis=1)

    momentum = zscore_col(df, sec, cnt, "mom")

    df["value_score"] = value
    df["quality_score"] = quality
    df["momentum_score"] = momentum
    df["SCORE"] = WEIGHTS["value"] * value + WEIGHTS["quality"] * quality + WEIGHTS["momentum"] * momentum
    df = df.sort_values("SCORE", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    # ---- flag compliance MNPI (informativo, NON scritto nello schema dashboard) ----
    is_italy_fin = df["country"].str.lower().isin(ITALY_TOKENS) & df["sector"].str.lower().str.contains("financ", na=False)
    if is_italy_fin.any():
        log(f"  >> ATTENZIONE COMPLIANCE: {int(is_italy_fin.sum())} nomi finanziari italiani "
            f"richiedono self-check MNPI prima di qualunque trade (ruolo ECM): "
            f"{', '.join(df.loc[is_italy_fin, 'ticker'].tolist())}")

    # ---- output nello schema ESATTO gia' accettato dalla tab SCREEN ----
    # (earn_yield/ebit_yield/roic/op_margin/debt_eq volutamente omessi: qui
    # sarebbero forward P/E, EBITDA/EV e Net Debt/EBITDA, non le stesse
    # metriche del motore US -- mostrarle sotto le stesse etichette sarebbe
    # fuorviante. country e' extra: la dashboard la ignora, resta per uso tuo.)
    out_cols = ["rank", "sector", "SCORE", "value_score", "quality_score", "momentum_score",
                "fcf_yield", "roe", "mom", "price", "high_52w", "low_52w", "mktcap", "country"]
    out = df.set_index("ticker")[out_cols].round(4)
    out.to_csv(out_path)

    meta = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_names": int(len(out)),
        "universe": "Internazionale manuale (Europa, EM incl. EM-Asia, GCC ex-UAE, Giappone, Canada, Australia)",
        "weights": WEIGHTS,
        "source": f"Bloomberg EQS export ({in_path.name}) -- pipeline manuale, non su GitHub Actions",
        "note": "earn_yield/ebit_yield/roic/op_margin/debt_eq omessi di proposito: le etichette fisse della "
                "dashboard non corrisponderebbero alle metriche disponibili da EQS (P/E fwd, EV/EBITDA, "
                "Net Debt/EBITDA anziche' trailing earn yield, EBIT/EV, Debt/Equity).",
    }
    with open(OUT_DIR / "meta_intl.json", "w") as f:
        json.dump(meta, f, indent=2)

    log(f"\nFatto: {len(out)} nomi scritti in {out_path.name}")
    log("Top 10:")
    log(out.head(10)[["rank", "sector", "SCORE", "country"]].to_string())
    log(f"\nCarica {out_path.name} nella tab SCREEN col bottone 'Importa ranking.csv'.")


if __name__ == "__main__":
    main()
