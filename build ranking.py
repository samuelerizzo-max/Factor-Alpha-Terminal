#!/usr/bin/env python3
# ============================================================
# FACTOR ALPHA — motore di scoring (value + quality + momentum)
# Versione per GitHub Actions (deriva da motore_fase1_v4.py,
# validato su Colab). Stessa identica logica: TTM da EDGAR
# (metodo FY + YTD corrente - YTD anno prima), esclude
# Financials e Real Estate, cap sui ratio estremi, z-score
# neutrali per settore, penalita' sui fattori mancanti.
#
# Output: ranking.csv + meta.json (alla radice del repo)
# ============================================================
import json
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Personalizza con un contatto vero: la SEC lo richiede per l'accesso equo a EDGAR.
UA = {"User-Agent": "factor-alpha-terminal contact@example.com"}
OUT_DIR = Path(__file__).resolve().parent  # tutto alla radice del repo, niente sottocartelle
WEIGHTS = {"value": 0.40, "quality": 0.40, "momentum": 0.20}
MISSING_PENALTY = -0.75


def log(*a):
    print(*a, flush=True)


def d(s):
    try:
        return dt.date.fromisoformat(s)
    except Exception:
        return None


def facts_of(cik):
    try:
        return requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers=UA, timeout=60,
        ).json().get("facts", {})
    except Exception:
        return {}


def durations(gaap, tags):
    for t in tags:
        node = gaap.get(t)
        if not node:
            continue
        rows = []
        for unit, arr in node["units"].items():
            if unit != "USD":
                continue
            for r in arr:
                if r.get("start") and r.get("end") and r.get("form") in ("10-K", "10-Q"):
                    a, b = d(r["start"]), d(r["end"])
                    if a and b:
                        rows.append({"end": r["end"], "val": r["val"], "days": (b - a).days})
        if rows:
            return rows
    return []


def instant(gaap, tags):
    for t in tags:
        node = gaap.get(t)
        if not node:
            continue
        best = None
        for unit, arr in node["units"].items():
            if unit not in ("USD", "shares"):
                continue
            for r in arr:
                if r.get("end") and r.get("form") in ("10-K", "10-Q"):
                    if best is None or r["end"] > best["end"]:
                        best = r
        if best:
            return best["val"]
    return None


def ttm(gaap, tags):
    rows = durations(gaap, tags)
    if not rows:
        return None
    annuals = [r for r in rows if 330 <= r["days"] <= 400]
    if not annuals:
        return None
    fy = max(annuals, key=lambda x: x["end"])
    latest_end = max(r["end"] for r in rows)
    if latest_end <= fy["end"]:
        return fy["val"]
    same = [r for r in rows if r["end"] == latest_end]
    yc = max(same, key=lambda x: x["days"])
    target = d(yc["end"]) - dt.timedelta(days=365)
    cand = [r for r in rows if abs((d(r["end"]) - target).days) <= 20 and abs(r["days"] - yc["days"]) <= 25]
    if not cand:
        return fy["val"]
    yp = min(cand, key=lambda x: abs((d(x["end"]) - target).days))
    return fy["val"] + yc["val"] - yp["val"]


REV = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
       "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"]
NI = ["NetIncomeLoss", "ProfitLoss"]
OPI = ["OperatingIncomeLoss"]
GP = ["GrossProfit"]
CFO = ["NetCashProvidedByUsedInOperatingActivities",
       "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
CAPEX = ["PaymentsToAcquirePropertyPlantAndEquipment"]
EQ = ["StockholdersEquity"]
ASSETS = ["Assets"]
DEBT = ["LongTermDebt", "LongTermDebtNoncurrent"]
CASH = ["CashAndCashEquivalentsAtCarryingValue"]
TAX = ["IncomeTaxExpenseBenefit"]
PRETAX = ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
          "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"]


def build_universe():
    sp = pd.read_csv(
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    )
    sp.columns = [c.strip() for c in sp.columns]
    sp = sp[~sp["GICS Sector"].isin(["Financials", "Real Estate"])].copy()
    return [(r["Symbol"], str(int(r["CIK"])).zfill(10), r["GICS Sector"]) for _, r in sp.iterrows()]


def fetch_fundamentals(universe):
    rows = []
    for i, (tk, cik, sec) in enumerate(universe):
        gaap = facts_of(cik).get("us-gaap", {})
        dei_facts = facts_of(cik).get("dei", {})
        if not gaap:
            continue
        rev, ni, opi, gp = ttm(gaap, REV), ttm(gaap, NI), ttm(gaap, OPI), ttm(gaap, GP)
        cfo, capex = ttm(gaap, CFO), ttm(gaap, CAPEX)
        tax, pretax = ttm(gaap, TAX), ttm(gaap, PRETAX)
        eq, assets = instant(gaap, EQ), instant(gaap, ASSETS)
        debt = instant(gaap, DEBT) or 0
        cash = instant(gaap, CASH) or 0
        shares = instant(dei_facts, ["EntityCommonStockSharesOutstanding"]) or instant(
            gaap, ["CommonStockSharesOutstanding"]
        )
        fcf = (cfo - capex) if (cfo is not None and capex is not None) else None
        taxrate = (tax / pretax) if (tax is not None and pretax not in (None, 0)) else 0.21
        taxrate = min(max(taxrate, 0), 0.5)
        nopat = opi * (1 - taxrate) if opi is not None else None
        invested = (eq + debt - cash) if eq is not None else None
        rows.append({
            "ticker": tk, "sector": sec, "rev": rev, "ni": ni, "opi": opi, "gp": gp,
            "fcf": fcf, "eq": eq, "assets": assets, "debt": debt, "cash": cash,
            "shares": shares, "nopat": nopat, "invested": invested,
        })
        if (i + 1) % 50 == 0:
            log(f"  ...{i + 1}/{len(universe)}")
        time.sleep(0.12)  # ~10 richieste/sec, cortesia verso i server SEC
    return rows


def zscore_col(df, sec, cnt, col, sign=1):
    x = pd.to_numeric(df[col], errors="coerce")
    x = x.clip(x.quantile(0.02), x.quantile(0.98))
    g = x.groupby(sec)
    zsec = (x - g.transform("mean")) / g.transform("std")
    zglob = (x - x.mean()) / x.std(ddof=0)
    z = zsec.where(cnt >= 6, zglob)
    return (z * sign).clip(-3, 3).fillna(MISSING_PENALTY)


def main():
    log("Costruisco l'universo S&P 500 ex-Financials/Real Estate...")
    universe = build_universe()
    log(f"universo: {len(universe)} aziende")

    log("Scarico i fondamentali da SEC EDGAR (TTM point-in-time)...")
    rows = fetch_fundamentals(universe)
    log(f"fondamentali raccolti: {len(rows)}")

    df = pd.DataFrame(rows).set_index("ticker")
    sector_map = df["sector"]

    log("Scarico prezzi e momentum da Yahoo Finance...")
    tickers = [t.replace(".", "-") for t in df.index]
    px = yf.download(tickers + ["SPY"], period="400d", progress=False, auto_adjust=True)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()

    def price_now(tk):
        c = tk.replace(".", "-")
        s = px[c].dropna() if c in px.columns else None
        return float(s.iloc[-1]) if (s is not None and len(s)) else None

    def mom_12_1(tk):
        c = tk.replace(".", "-")
        if c not in px.columns:
            return None
        s = px[c].dropna()
        return float(s.iloc[-21] / s.iloc[-252] - 1) if len(s) >= 260 else None

    def range_52w(tk):
        # stessa finestra di 252 giorni di borsa usata per il momentum,
        # cosi' "52 settimane" vuol dire la stessa cosa ovunque nel motore.
        c = tk.replace(".", "-")
        if c not in px.columns:
            return None, None
        s = px[c].dropna()
        window = s.iloc[-252:] if len(s) >= 252 else s
        if not len(window):
            return None, None
        return float(window.max()), float(window.min())

    df["price"] = [price_now(t) for t in df.index]
    df["mom"] = [mom_12_1(t) for t in df.index]
    ranges = [range_52w(t) for t in df.index]
    df["high_52w"] = [r[0] for r in ranges]
    df["low_52w"] = [r[1] for r in ranges]

    df["mktcap"] = df["price"] * df["shares"]
    df["ev"] = df["mktcap"] + df["debt"] - df["cash"]
    pos_eq = df["eq"] > 0
    df["earn_yield"] = df["ni"] / df["mktcap"]
    df["fcf_yield"] = df["fcf"] / df["mktcap"]
    df["ebit_yield"] = df["opi"] / df["ev"]
    df["book_price"] = np.where(pos_eq, df["eq"] / df["mktcap"], np.nan)
    df["roe"] = np.where(pos_eq, df["ni"] / df["eq"], np.nan)
    df["roic"] = np.where(df["invested"] > 0, df["nopat"] / df["invested"], np.nan)
    df["gross_margin"] = df["gp"] / df["rev"]
    df["op_margin"] = df["opi"] / df["rev"]
    df["debt_eq"] = np.where(pos_eq, df["debt"] / df["eq"], np.nan)
    df = df[(df["mktcap"] > 0) & (df["rev"].fillna(0) > 0)]

    # sanity: cappa i ratio impossibili prima dello z-score
    df["op_margin"] = df["op_margin"].clip(-1.0, 1.0)
    df["gross_margin"] = df["gross_margin"].clip(-1.0, 1.0)
    df["earn_yield"] = df["earn_yield"].clip(-0.25, 0.25)
    df["fcf_yield"] = df["fcf_yield"].clip(-0.25, 0.25)
    df["ebit_yield"] = df["ebit_yield"].clip(-0.30, 0.30)
    df["book_price"] = df["book_price"].clip(0, 3)
    df["roic"] = df["roic"].clip(-0.5, 0.60)
    df["roe"] = df["roe"].clip(-1.0, 0.80)
    df["debt_eq"] = df["debt_eq"].clip(0, 5)

    sec = sector_map.loc[df.index]
    cnt = sec.groupby(sec).transform("count")

    value = pd.concat([
        zscore_col(df, sec, cnt, "earn_yield"), zscore_col(df, sec, cnt, "fcf_yield"),
        zscore_col(df, sec, cnt, "ebit_yield"), zscore_col(df, sec, cnt, "book_price"),
    ], axis=1).mean(axis=1)
    quality = pd.concat([
        zscore_col(df, sec, cnt, "roic"), zscore_col(df, sec, cnt, "roe"),
        zscore_col(df, sec, cnt, "op_margin"), zscore_col(df, sec, cnt, "gross_margin"),
        zscore_col(df, sec, cnt, "debt_eq", sign=-1),
    ], axis=1).mean(axis=1)
    momentum = zscore_col(df, sec, cnt, "mom")

    df["value_score"] = value
    df["quality_score"] = quality
    df["momentum_score"] = momentum
    df["SCORE"] = WEIGHTS["value"] * value + WEIGHTS["quality"] * quality + WEIGHTS["momentum"] * momentum
    df = df.sort_values("SCORE", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    cols = ["rank", "sector", "SCORE", "value_score", "quality_score", "momentum_score",
            "earn_yield", "fcf_yield", "ebit_yield", "roic", "roe", "op_margin",
            "debt_eq", "mom", "price", "high_52w", "low_52w", "mktcap"]
    out = df[cols].round(4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "ranking.csv")

    meta = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "n_names": int(len(out)),
        "universe": "S&P 500 ex-Financials/Real Estate",
        "weights": WEIGHTS,
        "source": "SEC EDGAR (TTM point-in-time) + Yahoo Finance",
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log(f"\nFatto: {len(out)} nomi scritti in ranking.csv")
    log("Top 10:")
    log(out.head(10)[["rank", "sector", "SCORE"]].to_string())


if __name__ == "__main__":
    main()
