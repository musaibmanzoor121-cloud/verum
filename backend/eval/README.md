# Verum — Evaluation Harness

A small, dependency-free way to **measure** how well Engine A (the AI-text
detector) actually separates human writing from AI-generated writing — and to
report those numbers honestly, false positives and all.

## Why this exists

It is easy to *claim* "99% accurate." It is honest to *measure*. This harness
scores a hand-labeled set of resume/cover-letter snippets with the exact same
code the live API uses, then prints a confusion matrix and the standard
metrics. The number that matters most for a hiring tool — the **false-positive
rate** (a real person wrongly flagged) — is printed on every table.

## Run it

From the `backend/` folder, in your normal virtual environment (the one that
runs the app), no extra installs needed:

```bash
cd backend
python -m eval.evaluate
```

This prints the report to your terminal and writes `eval/REPORT.md`.

## What's in here

| File          | What it does                                                        |
|---------------|---------------------------------------------------------------------|
| `dataset.py`  | ~28 hand-labeled samples (14 human, 14 AI), with intentional hard cases |
| `metrics.py`  | Pure-Python confusion matrix, accuracy, precision, recall, specificity, FPR, F1, ROC-AUC |
| `evaluate.py` | Scores every sample, picks operating thresholds, writes `REPORT.md` |
| `REPORT.md`   | The generated results (regenerate any time by re-running)           |

## How to read the numbers

- **ROC-AUC** is the headline. It is threshold-independent — the probability
  that a random AI sample scores higher than a random human one. It can't be
  gamed by cherry-picking a cutoff. 1.0 is perfect, 0.5 is a coin flip.
- **False-positive rate** is the fairness metric. Verum is tuned to keep this
  low even if it means missing some AI, because wrongly accusing an honest
  applicant is the worst outcome for a screening tool.
- The report shows the mistakes **by sample id** so you can see exactly which
  cases are hard and judge for yourself whether the detector is behaving
  sensibly.

## Honesty notes

The dataset intentionally contains overlapping hard cases — polished humans who
use em-dashes and rule-of-three phrasing, and lightly-humanized AI where the
obvious tells were edited out. That is why the measured accuracy is **not**
100%. A detector that scores 100% on its own test set is usually overfit or
tested on easy examples.

To publish a defensible production number, expand `dataset.py` with real,
**consented** samples, keep the two classes balanced, and re-run. Treat the
current figures as a sanity check and a regression guard, not a guarantee.
