<div align="center">

# Verum

**Applicant Authenticity Engine**

Explainable screening for the age of AI-generated applications.
Verum flags AI-written resumes and bot-submitted applications — and *shows its evidence* for every call it makes.

[![CI](https://github.com/musaibmanzoor121-cloud/verum/actions/workflows/ci.yml/badge.svg)](https://github.com/musaibmanzoor121-cloud/verum/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](./LICENSE)

[Live demo](#live-demo) · [How it works](#how-it-works) · [Quick start](#quick-start) · [API](#api-reference) · [Configuration](#configuration)

</div>

---

## Table of Contents

- [What is Verum?](#what-is-verum)
- [Live demo](#live-demo)
- [Why explainability is the whole point](#why-explainability-is-the-whole-point)
- [How it works](#how-it-works)
  - [Engine A — Content authenticity](#engine-a--content-authenticity)
  - [Learned weights (optional ML upgrade)](#learned-weights-optional-ml-upgrade)
  - [Engine B — Submission authenticity](#engine-b--submission-authenticity)
  - [Cross-document consistency](#cross-document-consistency)
  - [Fusion](#fusion)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Testing](#testing)
- [Configuration](#configuration)
- [Engineering decisions](#engineering-decisions)
- [Privacy & ethics](#privacy--ethics)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## What is Verum?

Recruiters now receive a flood of applications where the resume was written by ChatGPT and the form was filled out by a script. Most "AI detectors" answer this with a single scary number — *"87% AI"* — that nobody can trust, explain, or appeal.

**Verum takes a different stance.** It runs two independent detection engines over each application and, instead of a verdict-in-a-black-box, it returns a **confidence band** plus a **receipt of evidence**: which sentences were unusually uniform, which buzzwords were stacked, whether the cover letter claims skills the resume never mentions, whether the form was submitted in 0.4 seconds by something that never moved a mouse.

> **Design principle:** Verum *never auto-rejects a candidate.* It routes the uncertain ones to a human with the evidence attached. A false positive should cost a second look — never someone's job opportunity.

---

## Live demo

> **Live app → [verum-zcf0.onrender.com](https://verum-zcf0.onrender.com)**
>
> _Hosted on Render's free tier — if the app has been idle it may take ~30–50 seconds to wake on the first request, then it's fast._

| Applicant view | Recruiter console |
| --- | --- |
| ![Applicant view](docs/screenshot-applicant.png) | ![Recruiter console](docs/screenshot-recruiter.png) |

Try it in seconds: the applicant form ships with **"Load human-written sample"** and **"Load AI-written sample"** buttons, so a reviewer can watch the two engines light up differently without typing a word.

---

## Why explainability is the whole point

| Typical AI detector | Verum |
| --- | --- |
| One number: *"87% AI"* | A **band**: *"68–88% likely AI-assisted"* — honest about its own uncertainty |
| No reasons given | An **evidence feed**: every flag names the signal, explains it in plain English, and highlights the exact text |
| Auto-rejects | **Routes to human review** — accessibility- and fairness-first |
| Text only | **Text + behavior + cross-document consistency**, fused into one report |

Fake precision is the enemy. A model that says "87.3%" is lying about its resolution; Verum reports a range because that is what the signals actually support.

---

## How it works

Verum fuses **three analyzers** into a single `AuthenticityReport`.

### Engine A — Content authenticity

`services/ai_analyzer.py` — detects AI-generated *writing* from several cheap, independent signals, no paid API required. It analyzes the résumé **and** the cover letter together, because flowing cover-letter prose is far more revealing than terse résumé bullets.

| Signal | What it measures | Why it flags AI |
| --- | --- | --- |
| **AI cadence phrases** | Stock phrasings current chat models overuse ("a testament to", "thrive in fast-paced environments", "not only… but also") | The strongest cheap tell against a modern LLM |
| **Buzzword / filler density** | Rate of generic filler ("results-driven", "leveraged", "passionate") per 100 words | AI-written applications stack corporate filler |
| **Em-dash character** | Use of the typographic "—" / "–" | People type a plain hyphen "-"; the em-dash *character* is a machine trait |
| **Triadic "rule of three"** | Parallel "X, Y, and Z" lists | Chat models lean on triads far more than most human writers |
| **Repeated openers** | Sentences starting with the same transition word | LLMs over-use "Furthermore / Moreover / Additionally" |
| **Burstiness** | Coefficient of variation of sentence lengths | Older models wrote uniformly — kept as a low-weight tie-breaker, since modern models vary sentence length on purpose |
| **Perplexity** *(optional)* | GPT-2 surprise score of the text | Low perplexity ≈ predictable ≈ machine-written |

> Modern frontier models defeat sentence-length ("burstiness") detection, so Engine A leans on **lexical and structural** tells — cadence phrasing, em-dash characters and triads — with burstiness as a minor tie-breaker. Perplexity uses HuggingFace `transformers` and is **off by default** so the app stays fast and deployable on a free tier — see [Configuration](#configuration).

### Learned weights (optional ML upgrade)

The seven signals above are combined into one score. By default that combination uses **hand-tuned weights**. But those same seven features can also feed a **logistic-regression classifier whose weights are *learned* from the labeled dataset** — turning Engine A from a rule-of-thumb into a small, honest ML model. It's off by default (`USE_LEARNED_MODEL=true` to enable) and, if the trained file is ever missing, Engine A silently falls back to the heuristic.

Two deliberate design choices make this more than a checkbox:

**The learned model independently *confirms* the hand-tuning.** Because it's trained on the raw `0..1` features (no standardization), each learned coefficient lands on the same scale as the constant it replaces — so the two are directly comparable:

| Feature | Learned coef | Hand-picked | |
| --- | --- | --- | --- |
| `buzzword` density | **+1.54** | 0.28 | strongest tell in both |
| `cadence` phrases | **+1.02** | 0.24 | second in both |
| `distinct_signals` (corroboration) | +0.82 | — | several tells firing at once matters |
| `triadic` "rule of three" | +0.38 | 0.15 | |
| `burstiness` | +0.34 | 0.08 | |
| `em_dash` character | +0.18 | 0.16 | |
| `repeated openers` | +0.00 | 0.09 | learned to ignore it (redundant with cadence) |

Trained from scratch with no knowledge of the heuristic, the model ranks **buzzword** and **cadence** as the top two tells — exactly what a human weighted highest by hand — and zeroes out repeated openers as redundant. That agreement is evidence the hand-tuned engine was capturing something real.

**The reported accuracy is honest, not inflated.** Performance is measured by **leave-one-out cross-validation** — for each of the 28 samples the model is trained on the *other* 27 and then scored on the one it never saw, so no sample grades its own homework:

| | ROC-AUC | False-positive rate | Recall |
| --- | --- | --- | --- |
| Heuristic (default) | **0.90** | **0%** | 79% |
| Learned (cross-validated) | 0.84 | 7% | 79% |

The learned model is competitive but does **not** beat the domain-tuned heuristic on this tiny set — and on a hiring tool a 0% false-positive rate matters more than a marginal AUC — which is exactly why the **heuristic stays the shipped default** and the learned model is opt-in. Every score stays explainable: the model reports its top signed contributions ("biggest factors: buzzword +0.31, cadence +0.22…"), so the recruiter view never loses its "why." The whole pipeline scales — point the trainer at a larger labeled corpus and retrain with no code change. Full write-up, per-sample scores, and limitations in the auto-generated [`backend/eval/MODEL_CARD.md`](backend/eval/MODEL_CARD.md).

### Engine B — Submission authenticity

`services/behavior_scorer.py` + `middleware/bot_detector.py` — detects *bots and scripts* from how the form was submitted, not from IP blocklists.

| Signal | What it measures |
| --- | --- |
| **Honeypot** | A hidden field no human can see; if it's filled, it's a bot |
| **Time-to-submit** | Sub-second submissions are non-human |
| **User-agent** | Flags `curl`, `python-requests`, `headless`, `selenium`, etc. |
| **Mouse entropy** | Shannon entropy of cursor movement angles — bots move in straight lines |
| **Keystroke rhythm** | Coefficient of variation of inter-key timings — bots type with metronomic regularity |
| **Rate limiting** | In-memory sliding window per IP (429 on abuse) |

> Behavioral signals capture **timing and geometry only** — never *what* you typed. See [Privacy & ethics](#privacy--ethics).

### Cross-document consistency

`services/consistency_checker.py` — compares the resume against the cover letter and flags skills or named entities (companies, tools) the cover letter claims but the resume never backs up — a classic "generated a cover letter separately" tell.

### Fusion

`main.py` fuses the three engines. A naïve weighted average has a dangerous blind spot: when a **real person** submits AI-written text, the bot score is legitimately near zero and would drag the average below the review line — hiding obvious AI writing. But these are *independent* concerns: AI-written text is worth a second look even if a human clicked submit. So the overall concern is the **higher** of a weighted blend and the strongest single engine:

```
blend   = 0.45 · bot  +  0.40 · ai_content  +  0.15 · inconsistency
overall = max( blend,  0.85 · ai_content,  0.90 · bot )
```

mapped to a verdict:

| Score | Verdict |
| --- | --- |
| `< 0.34` | Likely authentic |
| `0.34 – 0.66` | Needs human review |
| `> 0.66` | Strong AI/automation signals |

The floor is deliberately scaled (×0.85 / ×0.90), so a single engine has to be fairly confident to raise the alarm alone — which keeps genuine applicants from being flagged by one weak signal.

---

## Architecture

```mermaid
flowchart LR
    subgraph Browser["React SPA (applicant)"]
        F["Apply form<br/>+ hidden honeypot"]
        T["behaviorTracker.js<br/>mouse · keystroke timing"]
    end
    F -->|resume + cover letter + files| API
    T -->|behavior payload| API

    subgraph Server["FastAPI service (one URL)"]
        MW["BotDetection middleware<br/>rate limit · UA capture"]
        API["/api/analyze/"]
        EA["Engine A<br/>content"]
        EB["Engine B<br/>behavior"]
        EC["Consistency<br/>checker"]
        R["AuthenticityReport<br/>bands + evidence"]
        MW --> API --> EA & EB & EC --> R
    end

    R -->|JSON| DASH["Recruiter console<br/>bands · evidence receipts"]
    API -. serves built SPA .-> Browser
```

The built React app is served **by the FastAPI process itself** from `/static`, so the whole product runs as **one service at one URL** — no CORS in production, no second deployment.

---

## Tech stack

**Backend** — Python 3.11 · FastAPI · Pydantic v2 · Starlette middleware · pdfplumber · python-docx · (optional) PyTorch + Transformers
**ML** — logistic regression over 7 stylometric features · numpy trainer (dev/CI only) · **pure-Python serving** (zero runtime ML deps) · leave-one-out cross-validation
**Frontend** — React 18 · Vite 5 · Tailwind CSS 3.4
**Infra** — Docker (multi-stage) · docker-compose · Render blueprint
**Tests & CI** — pytest · GitHub Actions (unit tests + a quality gate + honest evals run on every push, across Python 3.10/3.11/3.12)

---

## Project structure

```
verum/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, endpoints, report fusion, SPA hosting
│   │   ├── config.py               # env-driven settings (all knobs live here)
│   │   ├── schemas.py              # Pydantic request/response models
│   │   ├── services/
│   │   │   ├── ai_analyzer.py      # Engine A — content authenticity
│   │   │   ├── features.py         # text → 7 stylometric features (shared by heuristic + model)
│   │   │   ├── ml_model.py         # pure-Python serving of the learned weights
│   │   │   ├── model.json          # learned logistic-regression weights (trained artifact)
│   │   │   ├── behavior_scorer.py  # Engine B — bot/behavior scoring
│   │   │   ├── consistency_checker.py  # resume ↔ cover-letter cross-check
│   │   │   └── pdf_parser.py       # PDF/DOCX text + metadata extraction
│   │   └── middleware/
│   │       └── bot_detector.py     # rate limiting + UA capture
│   ├── tests/
│   │   ├── test_engines.py         # engine unit tests
│   │   ├── test_eval_gate.py       # CI quality gate — heuristic ROC-AUC + fairness guard
│   │   └── test_learned_gate.py    # CI quality gate — learned model (cross-validated) guard
│   ├── eval/                       # honest, reproducible performance harness
│   │   ├── dataset.py              # 28 hand-labeled human/AI samples (incl. hard cases)
│   │   ├── metrics.py              # pure-Python confusion matrix, ROC-AUC, threshold sweep
│   │   ├── evaluate.py             # accuracy / ROC-AUC    → eval/REPORT.md
│   │   ├── robustness.py           # adversarial evasion   → eval/ROBUSTNESS.md
│   │   ├── fairness.py             # per-writing-style FPR → eval/FAIRNESS.md
│   │   └── train_model.py          # trains the learned model → model.json + MODEL_CARD.md
│   ├── conftest.py                 # makes `app`/`eval` importable under pytest
│   ├── requirements.txt
│   └── requirements-dev.txt        # pytest + httpx + numpy (dev/CI-only deps)
├── frontend/
│   └── src/
│       ├── App.jsx                 # applicant / recruiter tab shell
│       ├── api.js                  # API client (same-origin in prod)
│       ├── components/             # ApplyForm, Dashboard, ScoreBand, EvidenceCard, EnginePanel
│       └── lib/                    # behaviorTracker.js, ui.js
├── .github/workflows/ci.yml        # tests + quality gate + evals on every push
├── Dockerfile                      # multi-stage: build UI → serve from API
├── docker-compose.yml
├── render.yaml                     # one-click Render blueprint
└── README.md
```

---

## Quick start

### Option A — Docker (one command)

```bash
docker compose up --build
# open http://localhost:8000
```

### Option B — Run locally for development

**1. Backend** (terminal 1)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload    # http://localhost:8000
```

**2. Frontend** (terminal 2)

```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

In dev, the React app talks to the API at `http://localhost:8000`. In production, they're served from the same origin.

---

## API reference

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `POST` | `/api/analyze-text` | Analyze pasted resume/cover-letter text (JSON) |
| `POST` | `/api/analyze` | Analyze uploaded files + behavior (multipart) |

### Example — analyze text

```bash
curl -X POST http://localhost:8000/api/analyze-text \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "I am a highly results-driven and detail-oriented professional...",
    "cover_letter_text": "At Google I leveraged TensorFlow to spearhead ML initiatives...",
    "behavior": { "form_load_time": 0, "submit_time": 9000, "honeypot": "" }
  }'
```

### Example response *(abridged, real output)*

```json
{
  "verdict": "Needs human review",
  "overall_flag_score": 0.51,
  "content": {
    "ai_likelihood_band": "68–88% likely AI-assisted",
    "ai_likelihood_score": 0.782,
    "burstiness": 0.075,
    "buzzword_density": 36.96,
    "evidence": [
      {
        "engine": "content",
        "signal": "low_burstiness",
        "detail": "Sentence lengths are unusually uniform (variation index 0.08; human writing is typically >0.45).",
        "severity": "high",
        "excerpt": "I am a highly results-driven and detail-oriented software professional."
      }
    ]
  },
  "consistency": {
    "consistency_score": 0.0,
    "mismatches": [
      "Skill 'tensorflow' is highlighted in the cover letter but not present in the resume.",
      "'At Google' is referenced in the cover letter but does not appear in the resume."
    ]
  },
  "behavior": {
    "bot_likelihood_band": "0–20% likely automated",
    "bot_likelihood_score": 0.104,
    "honeypot_triggered": false,
    "time_to_submit_seconds": 9.0
  },
  "limitations": "Verum surfaces signals, not proof. Always pair with human judgment; never auto-reject."
}
```

Every report ends with a `limitations` note — the honesty is built into the payload, not just the docs.

---

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

The suite feeds a known **human-written** sample and a known **AI-written** sample through each engine and asserts they separate as expected (AI content scores high, honeypot trips, sub-second submits flag, human-like behavior stays low).

### Measured performance (not a marketing number)

Engine A ships with an evaluation harness that scores a hand-labeled set of
resume/cover-letter snippets with the exact code the live API uses, then reports
the results honestly — false positives included:

```bash
cd backend
python -m eval.evaluate        # prints the report and writes eval/REPORT.md
```

On the bundled 28-sample set (14 human, 14 AI, with deliberately hard
overlapping cases):

| Metric | Value |
| --- | --- |
| ROC-AUC (threshold-independent) | **0.90** |
| Precision | **100%** |
| Recall | **79%** |
| Specificity | **100%** |
| **False-positive rate** (real people wrongly flagged) | **0%** |

The dataset intentionally includes polished humans who use em-dashes and
rule-of-three phrasing, plus lightly-humanized AI where the obvious tells were
edited out — so the score is **not** a suspicious 100%. Verum is tuned to keep
the false-positive rate low even at the cost of recall: for a hiring tool,
wrongly accusing an honest applicant is worse than missing an AI draft, and
every flag routes to a human anyway. See [`backend/eval/`](backend/eval/) for
the harness and [`backend/eval/REPORT.md`](backend/eval/REPORT.md) for the full
per-sample breakdown.

### Train the learned model yourself

The optional learned classifier ([above](#learned-weights-optional-ml-upgrade))
is reproduced from the labeled data with one command:

```bash
cd backend
pip install -r requirements-dev.txt   # brings in numpy (training only)
python -m eval.train_model
```

That writes three artifacts: the servable weights
(`app/services/model.json`), the cross-validated per-sample scores
(`eval/cv_scores.json`), and a human-readable
[`eval/MODEL_CARD.md`](backend/eval/MODEL_CARD.md) with the learned-vs-hand-tuned
weight comparison and honest leave-one-out numbers. Serving never needs numpy —
that's a training-only dependency. Point `--data` at a larger labeled corpus to
scale up without touching code.

### Continuous integration

Every push and pull request runs the whole suite on GitHub Actions across Python
3.10, 3.11 and 3.12 (`.github/workflows/ci.yml`). CI **retrains the learned model
first** so the committed artifacts always reflect the current feature code, then
runs the unit tests plus **two quality gates** — one for the heuristic
(`backend/tests/test_eval_gate.py`) and one for the learned model
(`backend/tests/test_learned_gate.py`) — that fail the build if:

- ranking quality regresses (heuristic ROC-AUC below **0.85**, learned cross-validated ROC-AUC below **0.80**),
- any genuine human is flagged by the heuristic at the product threshold (**false positives > 0**),
- recall collapses (below **0.60**), or
- a **non-native-English** sample is ever flagged by either model — the bias guard described below.

So the accuracy claim can't silently rot: if a change makes the detector worse
or less fair, the badge turns red before it ships. All reports (including the
model card and cross-validation scores) are regenerated on every run and
uploaded as downloadable artifacts.

### Adversarial robustness — how easily can it be fooled?

`python -m eval.robustness` measures the honest ceiling of stylometric
detection: it takes each AI sample, applies realistic *humanizing* edits, and
reports how far the score drops. A determined applicant who strips the surface
tells (em-dash character, rule-of-three lists, stock phrasings, buzzwords)
pushes **9 of 14** AI samples back under the review line.

| "Humanizing" edit | Mean AI score | AI samples that now evade |
| --- | --- | --- |
| _(none — baseline)_ | 0.61 | 3 / 14 |
| Soften stock AI phrasings | 0.45 | 7 / 14 |
| All edits at once | 0.33 | 9 / 14 |

This isn't a bug — *every* content-only detector can be edited around, which is
exactly why Verum never auto-rejects and fuses content with submission-behavior
and cross-document signals that reworded prose can't touch. Full table in
[`backend/eval/ROBUSTNESS.md`](backend/eval/ROBUSTNESS.md).

### Fairness — who gets wrongly flagged?

AI detectors are documented to disproportionately flag **non-native English
writers** (Stanford, 2023). `python -m eval.fairness` buckets the human samples
by writing style and reports the false-positive rate per group:

| Human writing style | Mean score | False-positive rate |
| --- | --- | --- |
| Non-native English | 0.04 | **0%** |
| Formal / academic | 0.07 | **0%** |
| Early-career / buzzword-prone | 0.17 | **0%** |
| Polished (AI-overlapping) | 0.33 | **0%** |

No group is disproportionately flagged, and the non-native-English case is
pinned in place by the CI gate above. Full report in
[`backend/eval/FAIRNESS.md`](backend/eval/FAIRNESS.md).

---

## Configuration

Every setting is read from an environment variable with a sensible default (see `backend/app/config.py`), so you can change behavior in deployment without touching code.

| Variable | Default | Purpose |
| --- | --- | --- |
| `USE_TRANSFORMER_PERPLEXITY` | `false` | Load GPT-2 to compute real perplexity. Heavy (~500 MB + torch); off by default so the app fits a free tier. When off, a fast statistical proxy is used instead. |
| `PERPLEXITY_MODEL` | `gpt2` | HuggingFace model used when the transformer path is enabled. |
| `USE_LEARNED_MODEL` | `false` | Score Engine A with the **learned** logistic-regression weights (`app/services/model.json`) instead of the hand-tuned heuristic. Pure-Python, no extra deps. Off by default; falls back to the heuristic automatically if the model file is missing. See [Learned weights](#learned-weights-optional-ml-upgrade). |
| `MIN_SUBMIT_SECONDS` | `3.0` | Submissions faster than this (seconds) are treated as suspicious. |
| `RATE_LIMIT_MAX` | `30` | Max requests allowed per IP per window. |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Length of the rate-limit window, in seconds. |
| `CORS_ORIGINS` | `localhost:5173,3000,127.0.0.1:5173` | Comma-separated allowed frontend origins (dev only; production is same-origin). |
| `REDIS_URL` | *(empty)* | Optional Redis URL to back rate limiting for multi-instance deploys. In-memory when empty. |

To enable the heavier model locally: un-comment `torch` and `transformers` in `backend/requirements.txt`, copy `backend/.env.example` to `backend/.env`, set `USE_TRANSFORMER_PERPLEXITY=true`, and restart the backend.

---

## Engineering decisions

**Bands, not fake-precise scores.** A detector that reports "87.3%" implies a resolution it does not have. Verum reports a ±10 band so the output is honest about its own uncertainty.

**Graceful degradation.** The heavy GPT-2 perplexity model is optional and off by default. The app runs in seconds on a 512 MB free tier using pure-Python stylometry; if you have the resources, `USE_TRANSFORMER_PERPLEXITY=true` upgrades Engine A in place — same interface, richer signal.

**One service, one URL.** Rather than deploy an API and a static site separately, the Dockerfile builds the React bundle and lets FastAPI serve it. Simpler ops, no production CORS, one thing to monitor.

**Evidence is a first-class object.** Every signal emits an `EvidenceItem(engine, signal, detail, severity, excerpt)`. The UI is just a renderer over that list — which means the "why" can never drift from the "what".

**Human-in-the-loop by construction.** There is no code path that rejects an applicant. The strongest verdict Verum can reach is *"route to a human with this evidence."*

---

## Privacy & ethics

- Behavioral tracking records **timing and cursor geometry only** — never keystroke *content*.
- Submissions live **in memory** for the session; nothing is persisted to a database by default.
- The honeypot field is invisible to humans and to screen-reader users (positioned off-screen with `aria-hidden` / `tabindex=-1`).
- **No candidate is ever auto-rejected.** Flags are decision-support for a human reviewer.

---

## Limitations

Verum surfaces *signals*, not proof. Skilled writers can trip the buzzword detector; strong AI text can read as bursty; a careful bot can fake human-ish timing. These heuristics are meant to **prioritize human attention**, not to deliver a verdict. Treat every flag as "worth a closer look," never as grounds for rejection.

---

## Roadmap

- Redis-backed rate limiting for multi-instance deploys (interface already stubbed in `config.py`)
- Optional persistence layer (Postgres) for an audit trail
- Train the learned model on a large public corpus (e.g. HC3) so it can beat the heuristic, not just match it — the trainer already accepts `--data`
- PDF layout/font forensics for template detection

---

## License

[MIT](./LICENSE) — free to use, learn from, and build on.
