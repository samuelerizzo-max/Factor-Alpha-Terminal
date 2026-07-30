#!/usr/bin/env python3
# ============================================================
# INVIO EMAIL — Top 20 del ranking, dopo ogni aggiornamento riuscito
# ============================================================
# Non blocca mai il workflow principale: se qualcosa va storto qui
# (password non ancora configurata, problema SMTP), stampa l'errore
# e esce senza sollevare eccezione -- il ranking.csv resta comunque
# generato e committato, che e' la parte che conta di piu'.
# ============================================================
import os
import smtplib
import ssl
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
GMAIL_ADDRESS = "siqilliya.official@gmail.com"
GMAIL_TO = "samuele.rizzo@me.com"


def build_html_table(df):
    rows = ""
    for tk, r in df.iterrows():
        flag = r.get("trap_flag", "")
        flag = "" if pd.isna(flag) else flag
        color = "#e05260" if flag == "ALTO" else ("#2fa876" if flag == "BASSO" else "#888")
        fscore = f"{int(r['fscore'])}/9" if pd.notna(r.get("fscore")) else "—"
        rows += (
            f'<tr>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #333;">{int(r["rank"])}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #333;font-weight:600;">{tk}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #333;color:#999;font-size:12px;">{r["sector"]}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #333;">{r["SCORE"]:.3f}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #333;">{fscore}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #333;color:{color};">{flag or "—"}</td>'
            f'</tr>'
        )
    return (
        '<table style="border-collapse:collapse;font-family:monospace;font-size:13px;width:100%;">'
        '<tr style="background:#111;color:#aaa;text-align:left;">'
        '<th style="padding:6px 10px;">#</th><th style="padding:6px 10px;">Ticker</th>'
        '<th style="padding:6px 10px;">Settore</th><th style="padding:6px 10px;">Score</th>'
        '<th style="padding:6px 10px;">F-Score</th><th style="padding:6px 10px;">Flag</th>'
        '</tr>' + rows + '</table>'
    )


def main():
    ranking_path = OUT_DIR / "ranking.csv"
    meta_path = OUT_DIR / "meta.json"
    if not ranking_path.exists():
        print("ranking.csv non trovato, salto l'invio email.")
        return

    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not app_password:
        print("GMAIL_APP_PASSWORD non impostata (Secret mancante) -- salto l'invio email.")
        return

    df = pd.read_csv(ranking_path).set_index("ticker")
    top20 = df.sort_values("rank").head(20)

    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass
    generated_at = meta.get("generated_at", "n/d")
    n_names = meta.get("n_names", len(df))
    date_label = generated_at[:10] if generated_at != "n/d" else "oggi"

    html = f"""<html><body style="background:#06070A;color:#eee;padding:20px;">
      <h2 style="color:#F2A93B;">Factor Alpha Terminal — Top 20 di oggi</h2>
      <p style="color:#999;font-size:12px;">Generato: {generated_at} · Universo: {n_names} aziende (S&amp;P 500 ex-Financials/Real Estate)</p>
      {build_html_table(top20)}
      <p style="color:#666;font-size:11px;margin-top:20px;">
        Promemoria: lo SCORE e il ranking sono un punto di partenza per la selezione,
        non un segnale di acquisto. Nessuna posizione senza passare dai 5 Gate e da una
        tesi bear/base/bull scritta.
      </p>
    </body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Factor Alpha Terminal — Top 20 ({date_label})"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = GMAIL_TO
    msg.attach(MIMEText(html, "html"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls(context=context)
            server.login(GMAIL_ADDRESS, app_password)
            server.sendmail(GMAIL_ADDRESS, GMAIL_TO, msg.as_string())
        print(f"Email inviata a {GMAIL_TO} con la top 20.")
    except Exception as e:
        print(f"Invio email fallito (il ranking resta comunque generato): {e}")


if __name__ == "__main__":
    main()
