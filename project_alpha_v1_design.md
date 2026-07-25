# Project Alpha v1 — Documento di Design ("La Costituzione")

Versione: 0.1 (bozza iniziale, Weekend 0)
Autore: Sam + Claude
Stato: DRAFT — ambiente di esecuzione confermato (MacBook Air, vedi sezione 2)
Obiettivo: allocazione di capitale reale (confermato) — non solo esercizio intellettuale

---

## 1. Scopo e Filosofia

Project Alpha è una piattaforma di ricerca quantitativa, non una singola strategia. L'obiettivo di v1 non è generare rendimento: è dimostrare che il framework non mente (no look-ahead, no survivorship bias, backtest riproducibili).

Regole non negoziabili:

- **Ipotesi prima del codice.** Ogni idea deve rispondere a: qual è l'ipotesi economica? Perché dovrebbe funzionare? In quali mercati? Come dimostriamo che non è fortuna? Se non c'è risposta, non si implementa.
- **Niente indicatori tecnici come punto di partenza** (RSI 30/70, MACD, EMA crossover isolati). Si parte da anomalie documentate in letteratura o da ipotesi economiche nuove e argomentabili.
- **Niente OR logico tra modelli con logiche opposte** ("execute if any model gives BUY" è vietato: se due modelli hanno tesi economiche diverse, vanno validati e pesati separatamente, non sommati).
- **Riproducibilità totale.** Stesso codice + stessi dati + stesso seed = stessi numeri, sempre. Versionamento obbligatorio di codice, config e dati.
- **Scetticismo come default.** Il primo istinto su ogni risultato è cercare di smontarlo (look-ahead, leakage, overfitting, survivorship, costi irrealistici). Solo ciò che sopravvive ai tentativi di smentita è candidato.

## 2. Ambiente di Esecuzione

**Confermato:** MacBook Air 13" (chip M5, CPU 10-core, GPU 10-core, SSD 1TB, 24GB RAM). Ambiente locale reale disponibile — si torna al piano originale, niente più adattamento via Colab per questo progetto:

- **Sviluppo:** VS Code + Claude Code (CLI), in locale sul MacBook.
- **Repository:** Git locale + GitHub, push/pull normale (non più solo commit via Colab/GitHub Actions).
- **Package management:** uv, ambiente virtuale locale, Python 3.12+.
- **Dati:** fetch diretto da Python locale (Yahoo Finance, FRED, provider a pagamento se necessario) — nessuno dei vincoli di rete del sandbox di questa chat.
- **Esecuzione backtest/ricerca:** locale, non più dipendente da Colab.

Nota: Factor Alpha Terminal resta sul pattern Colab/GitHub Actions esistente per continuità — migrarlo in locale è una decisione separata, non necessaria per Project Alpha.

Questa chat resta utile per: revisione codice, discussione architetturale, interpretazione risultati, debug — ma lo sviluppo primario si sposta su Claude Code nel repository locale sul MacBook.

## 3. Architettura del Sistema

```
Data Engine → Feature Engine → Research Engine → Validation Engine → Portfolio Engine → Paper Trading → Live Trading
```

Ogni modulo indipendente e testabile in isolamento. Il "bot" (execution) è l'ultimo pezzo, non il primo.

## 4. Stack Tecnico

- Python 3.12+
- Polars + DuckDB (preferiti a Pandas per performance/scala)
- VectorBT per ricerca/backtest vettoriale
- Parquet per storage dati
- Plotly per visualizzazione
- Ruff (lint) + Pytest (test automatici)
- Git per versionamento (config, seed, codice — tutto tracciato)

## 5. Struttura del Repository

```
project_alpha/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── data_engine/       (codice di fetch/pulizia dati — v. nota sotto)
├── features/
├── strategies/
├── backtests/
├── validation/
├── portfolio/
├── execution/
├── paper_trading/
├── notebooks/        (solo esplorazione, mai logica core)
├── tests/
└── docs/
    ├── project_alpha_v1_design.md   (questo file)
    ├── research_journal.md
    └── strategy_cards/
```

Nota (Fase 1): la bozza originale non prevedeva una cartella per il codice del Data Engine — `data/` è solo storage (raw/processed/external), non contiene moduli Python. Aggiunta `data_engine/` come package a livello radice, parallelo a `features/`, `strategies/`, ecc., per ospitare il codice di fetch, pulizia e validazione dei dati.

## 6. Pipeline di Ricerca

1. Formulazione ipotesi economica (Sam).
2. Specifica quantitativa dettagliata e non ambigua (es. "cross-sectional momentum, ranking mensile, universo S&P 500, rebalance 20gg, esclusione ADV < X, commissioni Y, walk-forward").
3. Implementazione codice (Claude Code, in locale nel repository).
4. Review: il codice implementa davvero l'ipotesi? (Sam)
5. Backtest.
6. Validazione statistica (sezione 7).
7. Decisione: Promossa / Scartata / Da approfondire → Research Journal.

## 7. Criteri di Validazione (checklist di promozione)

Una strategia entra in portafoglio solo se supera **tutti** i seguenti:

- [ ] No look-ahead bias
- [ ] No survivorship bias
- [ ] Commissioni e slippage realistici
- [ ] Walk-forward analysis (expanding window, purge gap)
- [ ] Test out-of-sample
- [ ] Stabilità su periodi e regimi di mercato diversi
- [ ] Nessuna ottimizzazione eccessiva dei parametri
- [ ] Significatività statistica (Deflated Sharpe Ratio, Probability of Backtest Overfitting, White's Reality Check dove applicabile)
- [ ] Minimo 15-20 anni di dati dove disponibili

Se fallisce anche un solo punto: non entra, punto.

## 8. Scheda Strategia (template — una per ogni idea testata)

```
Nome:
Ipotesi economica:
Mercato:
Timeframe:
Dati richiesti:
Feature:
Regole di ingresso:
Regole di uscita:
Gestione del rischio:
Risultati in-sample:
Risultati out-of-sample:
Robustezza:
Decisione: [Promossa / Scartata / Da approfondire]
Motivazione della decisione:
```

## 9. Research Journal

File `docs/research_journal.md`, append-only. Ogni voce (promossa o scartata):

```
## Ipotesi #N — [nome]
Data:
Perché pensavamo funzionasse:
Come l'abbiamo testata:
Risultati:
Perché è stata scartata / promossa:
```

## 10. Guardrail di Compliance (ECM / MNPI) — implicazioni del capitale reale

**Obiettivo confermato: allocazione di capitale reale.** Questo rende il checklist di sezione 7 vincolante senza eccezioni prima della Fase 6 — non un dettaglio procedurale.

### 10.1 Dati e contenuto
- Qualunque fattore "event-driven" o legato a flussi istituzionali usa **solo dati pubblici e già divulgati** (filing SEC/DFM, dataset accademici di anomalie documentate — es. Post-Earnings Announcement Drift, insider transactions da Form 4 pubblici).
- **Mai** spunti derivati dalla pipeline reale di deal ECM di Sam presso DIB.
- Self-check MNPI prima di qualunque azione su titoli finanziari italiani o coperti da Group Personal Trading Policy, come già in uso per Banca IFIS.

### 10.2 Sizing e conto
- Il progetto vive nella sleeve speculativa/di ricerca, **separata e capped** dal book value-contrarian core (10-15 posizioni). Nessun commingling di capitale o rischio tra i due.
- Se una strategia opera su crypto: IBKR non offre l'esecuzione sistematica necessaria (accesso limitato via Paxos, pochi asset, nessuna infrastruttura da quant). Servirebbe un conto exchange separato — che significa un nuovo obbligo di disclosure verso Group Compliance, non solo l'aggiornamento di quanto già dichiarato per IBKR.

### 10.3 Scope della Group Personal Trading Policy — verifica consigliata
Dalla copia in tuo possesso (maggio 2025), i divieti di Day Trading (7.3) e di trading speculativo a breve termine <3gg (7.2) sono scritti per le **Covered Securities** (azioni/sukuk DIB, parent, subsidiaries, affiliates) — non per titoli USA/crypto non correlati. Sulla base del solo testo di questo documento, una strategia sistematica ad alto turnover su SPY/US equities/crypto non rientra letteralmente in quel divieto.

Detto questo: molte banche regolate affiancano alla Covered Securities policy una politica più ampia di personal account dealing (pre-approvazione, holding period minimo) che copre **tutte** le operazioni personali del dipendente, non solo i titoli della banca. Prima di allocare capitale reale con turnover elevato, vale una verifica esplicita con Group Compliance — è esattamente il tipo di attività che una policy più ampia tende a colpire, e chiederlo prima costa molto meno che scoprirlo dopo.

## 11. Ruoli

- **Sam — Lead Quant Researcher:** ipotesi, decisioni di architettura, interpretazione risultati, red-team dei backtest, decisione finale di promozione.
- **Claude — supporto ingegneristico:** implementazione, test, pipeline dati, refactoring, revisione codice contro la specifica. Non propone strategie non richieste; esegue specifiche precise.

## 12. Roadmap

- **Fase 0 (Weekend 0):** repo GitHub, questo documento, struttura cartelle, decisione ambiente definitiva.
- **Fase 1:** framework di backtest minimo + sanity check Buy & Hold SPY (dati fetchati in locale via Python).
- **Fase 2:** replica di 2-3 anomalie documentate (momentum, mean reversion, PEAD) per validare che il framework non menta.
- **Fase 3:** validation engine completo (walk-forward, Monte Carlo, stress test).
- **Fase 4:** portfolio engine (combinazione strategie a bassa correlazione).
- **Fase 5:** paper trading.
- **Fase 6:** capitale reale, solo dopo: checklist di sezione 7 superata su tutti i punti, periodo di paper trading concluso, conferma scritta di Group Compliance sullo scope della personal trading policy (10.3), e pre-clearance dove applicabile. Nessuno di questi gate è opzionale dato l'obiettivo di allocazione reale.

---
*Documento vivo: ogni modifica architetturale rilevante va versionata con commit dedicato.*
