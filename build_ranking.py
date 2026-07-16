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
import io
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Personalizza con un contatto vero: la SEC lo richiede per l'accesso equo a EDGAR.
UA = {"User-Agent": "factor-alpha-terminal contact@example.com"}
# iShares blocca gli User-Agent non-browser su alcuni endpoint; qui serve uno vero.
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
IWV_HOLDINGS_URL = ("https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/"
                     "1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund")
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
OUT_DIR = Path(__file__).resolve().parent  # tutto alla radice del repo, niente sottocartelle
WEIGHTS = {"value": 0.40, "quality": 0.40, "momentum": 0.20}
MISSING_PENALTY = -0.75
# S&P 500 non ne aveva bisogno (tutti large cap per definizione); Russell 3000
# arriva fino ai micro-cap, dove i nomi "piu' economici" sono spesso solo i
# meno liquidi, non i piu' sottovalutati. $300M come pavimento -- alzalo a
# $500M se vuoi allinearlo esattamente alla soglia che usi negli screen EQS.
MIN_MKTCAP = 300_000_000


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


def annual_pair(gaap, tags, want_duration=True):
    """Valore piu' recente e valore dell'anno precedente, da depositi 10-K
    annuali soltanto (confronto anno su anno per il Piotroski F-Score).
    want_duration=True per voci di conto economico/cash flow (richiede un
    periodo di 330-400 giorni); False per voci di stato patrimoniale
    (istantanee, qualunque "start" non serve)."""
    for t in tags:
        node = gaap.get(t)
        if not node:
            continue
        rows = set()
        for unit, arr in node["units"].items():
            if unit not in ("USD", "shares"):
                continue
            for r in arr:
                if r.get("form") != "10-K" or not r.get("end"):
                    continue
                if want_duration:
                    if not r.get("start"):
                        continue
                    a, b = d(r["start"]), d(r["end"])
                    if not (a and b and 330 <= (b - a).days <= 400):
                        continue
                rows.add((r["end"], r["val"]))
        if rows:
            rows = sorted(rows, key=lambda x: x[0])
            if len(rows) >= 2:
                return rows[-1][1], rows[-2][1]
            return rows[-1][1], None
    return None, None


def compute_fscore(gaap):
    """Piotroski F-Score (0-9): traiettoria anno su anno, non livello.
    Restituisce (punteggio, numero di criteri effettivamente calcolabili)."""
    ni_t, ni_p = annual_pair(gaap, NI)
    rev_t, rev_p = annual_pair(gaap, REV)
    gp_t, gp_p = annual_pair(gaap, GP)
    cfo_t, cfo_p = annual_pair(gaap, CFO)
    assets_t, assets_p = annual_pair(gaap, ASSETS, want_duration=False)
    debt_t, debt_p = annual_pair(gaap, DEBT, want_duration=False)
    debt_t, debt_p = debt_t or 0, debt_p or 0
    ca_t, ca_p = annual_pair(gaap, CURR_ASSETS, want_duration=False)
    cl_t, cl_p = annual_pair(gaap, CURR_LIAB, want_duration=False)
    sh_t, sh_p = annual_pair(gaap, ["CommonStockSharesOutstanding", "CommonStockSharesIssued"], want_duration=False)

    pts = {}
    pts["roa_pos"] = 1 if (ni_t is not None and assets_t and ni_t / assets_t > 0) else 0
    pts["cfo_pos"] = 1 if (cfo_t is not None and cfo_t > 0) else 0
    pts["droa"] = (1 if (ni_t is not None and assets_t and ni_p is not None and assets_p
                         and (ni_t / assets_t) > (ni_p / assets_p))
                   else (0 if all(v is not None for v in [ni_t, assets_t, ni_p, assets_p]) else None))
    pts["accruals"] = 1 if (cfo_t is not None and ni_t is not None and cfo_t > ni_t) else 0
    pts["leverage"] = (1 if (assets_t and assets_p and (debt_t / assets_t) < (debt_p / assets_p))
                        else (0 if (assets_t and assets_p) else None))
    pts["liquidity"] = (1 if (ca_t is not None and cl_t and ca_p is not None and cl_p
                              and (ca_t / cl_t) > (ca_p / cl_p))
                         else (0 if all(v is not None for v in [ca_t, cl_t, ca_p, cl_p]) else None))
    pts["no_dilution"] = (1 if (sh_t is not None and sh_p is not None and sh_t <= sh_p)
                           else (0 if all(v is not None for v in [sh_t, sh_p]) else None))
    pts["gross_margin"] = (1 if (gp_t is not None and rev_t and gp_p is not None and rev_p
                                 and (gp_t / rev_t) > (gp_p / rev_p))
                            else (0 if all(v is not None for v in [gp_t, rev_t, gp_p, rev_p]) else None))
    pts["asset_turnover"] = (1 if (rev_t is not None and assets_t and rev_p is not None and assets_p
                                   and (rev_t / assets_t) > (rev_p / assets_p))
                              else (0 if all(v is not None for v in [rev_t, assets_t, rev_p, assets_p]) else None))

    valid = [v for v in pts.values() if v is not None]
    return (sum(valid) if valid else None), len(valid)


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
CURR_ASSETS = ["AssetsCurrent"]
CURR_LIAB = ["LiabilitiesCurrent"]
TAX = ["IncomeTaxExpenseBenefit"]
PRETAX = ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
          "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"]


def fetch_iwv_holdings():
    """Scarica le holding dell'ETF iShares Russell 3000 (IWV). FTSE Russell
    non pubblica gratis la lista dei componenti dell'indice; le holding
    dell'ETF che lo replica sono il modo standard (non ufficiale, ma
    ampiamente usato da chi fa ricerca quant) per averle gratis. Il file ha
    alcune righe di metadata (nome fondo, data, AUM) prima dell'header vero
    -- lo cerco dinamicamente invece di contare le righe a mano, cosi' non
    si rompe se iShares aggiunge/toglie una riga di metadata."""
    resp = requests.get(IWV_HOLDINGS_URL, headers=BROWSER_UA, timeout=60)
    resp.raise_for_status()
    lines = resp.text.splitlines()
    header_idx = next((i for i, l in enumerate(lines) if l.startswith("Ticker,")), None)
    if header_idx is None:
        raise RuntimeError("Non trovo la riga di header nel CSV di IWV -- iShares potrebbe "
                            "aver cambiato formato del file. Controllare manualmente.")
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), thousands=",")
    df.columns = [c.strip() for c in df.columns]
    return df


def build_universe():
    log("Scarico le holding di IWV (iShares Russell 3000 ETF)...")
    iwv = fetch_iwv_holdings()
    iwv = iwv[iwv["Asset Class"] == "Equity"].copy()
    iwv["Ticker"] = iwv["Ticker"].astype(str).str.strip().str.upper()
    iwv = iwv[~iwv["Ticker"].isin(["--", "", "NAN"])]
    iwv = iwv[~iwv["Sector"].isin(["Financials", "Real Estate", "Cash and/or Derivatives"])]
    iwv = iwv.drop_duplicates(subset="Ticker")
    log(f"  {len(iwv)} nomi equity ex-Financials/Real Estate nell'ETF")

    log("Scarico la mappa ticker -> CIK da SEC...")
    sec_map = requests.get(SEC_TICKERS_URL, headers=UA, timeout=60).json()
    ticker_to_cik = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in sec_map.values()}

    universe, unmatched = [], []
    for _, r in iwv.iterrows():
        cik = ticker_to_cik.get(r["Ticker"])
        if cik:
            universe.append((r["Ticker"], cik, r["Sector"]))
        else:
            unmatched.append(r["Ticker"])
    if unmatched:
        preview = ", ".join(unmatched[:15]) + (" ..." if len(unmatched) > 15 else "")
        log(f"  >> ATTENZIONE: {len(unmatched)} ticker senza CIK SEC corrispondente, esclusi "
            f"(spesso simboli multi-classe scritti diversamente tra iShares e SEC/Yahoo, "
            f"es. BRKB vs BRK-B): {preview}")
    return universe


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
        fscore, fscore_n = compute_fscore(gaap)
        rows.append({
            "ticker": tk, "sector": sec, "rev": rev, "ni": ni, "opi": opi, "gp": gp,
            "fcf": fcf, "eq": eq, "assets": assets, "debt": debt, "cash": cash,
            "shares": shares, "nopat": nopat, "invested": invested,
            "fscore": fscore, "fscore_n": fscore_n,
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
    log("Costruisco l'universo Russell 3000 ex-Financials/Real Estate...")
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
    n_before_floor = len(df)
    df = df[(df["mktcap"] >= MIN_MKTCAP) & (df["rev"].fillna(0) > 0)]
    log(f"  soglia liquidita' (mktcap >= ${MIN_MKTCAP/1e6:.0f}M): {len(df)}/{n_before_floor} nomi restano")

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

    # ---- bandierina "rischio value trap" — NON un segnale di acquisto ----
    # Alta solo se il titolo e' statisticamente economico (top terzo per
    # value_score) E la traiettoria dei fondamentali (F-Score) sta
    # peggiorando (<=3/9). Bassa se e' economico E la traiettoria conferma
    # (F-Score >=7/9). Altrimenti vuota: non e' abbastanza economico perche'
    # la domanda "trappola o occasione" si applichi.
    cheap_threshold = df["value_score"].quantile(0.67)
    is_cheap = df["value_score"] >= cheap_threshold
    has_reliable_fscore = df["fscore_n"] >= 6
    df["trap_flag"] = ""
    df.loc[is_cheap & has_reliable_fscore & (df["fscore"] <= 3), "trap_flag"] = "ALTO"
    df.loc[is_cheap & has_reliable_fscore & (df["fscore"] >= 7), "trap_flag"] = "BASSO"

    cols = ["rank", "sector", "SCORE", "value_score", "quality_score", "momentum_score",
            "earn_yield", "fcf_yield", "ebit_yield", "book_price", "roic", "roe", "op_margin",
            "gross_margin", "debt_eq", "mom", "price", "high_52w", "low_52w", "mktcap",
            "fscore", "fscore_n", "trap_flag"]
    out = df[cols].round(4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "ranking.csv")

    meta = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "n_names": int(len(out)),
        "universe": "Russell 3000 ex-Financials/Real Estate",
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
