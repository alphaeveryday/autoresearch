# autoresearch — Quant Lab profile

This is the Quant port of Karpathy autoresearch. The original GPU training
profile remains at the repository root. Use this file as the running agent's
program for the Quant profile, not as background reading.

## Setup

The owner has authorized the continuous Quant research loop. Use a dedicated
worktree and preserve the inherited study history; do not ask for confirmation
between routine experiments. Read `train.py`, the fixed `prepare.py`, the study's
protocol and the three installed scientific skills before proposing a change.

The caller supplies `QUANTLAB_REPO`, `QUANTLAB_HOME`, and `QUANTLAB_PROTOCOL`.
Use the Quant Lab locked Python environment. Do not change installed packages.
`QUANTLAB_HOME` must contain the inherited run registry and fixed B0 reference.

## Editable and fixed parts

- Edit only `train.py` for a strategy experiment. Under Harness² this same file
  is staged as `scripts/train.py`; that location is the editable candidate.
- Keep its trading settings fixed. Change the model, inputs or target under a
  written hypothesis with a parent, one change, rival explanation and falsifier.
- Do not edit `prepare.py`, the engine, assessment policy, costs, data or holdout.
- `alpha(d)` receives completed bars. Delay 1 means a close T+1 assumed fill and
  the first earned return at T+2. Only matured labels may enter training.
- No submission, deployment or orders. The current engine has an unresolved
  future-bar-dependent position rule; outputs are development diagnostics and
  `promotion_eligible` remains false.

Before a hypothesis, read `.agents/skills/hypothesis-generation/SKILL.md` and its
Quant reference. Before interpreting a result, read
`.agents/skills/scientific-critical-thinking/SKILL.md` and
`.agents/skills/statistical-analysis/SKILL.md`, including their Quant references.

## Experiment loop

1. Start with an unchanged baseline. Preserve its source and previous failures.
2. Write the hypothesis and parent before changing the candidate. State which
   development observations informed it. Do not call it a new untouched test.
3. Make one candidate change; run its causal/synthetic checks first.
4. From this profile directory run the fixed evaluator (use the locked Python):
   `python prepare.py --candidate train.py --hypothesis "mechanism; rival; falsifier" --output /absolute/new-attempt`
5. Read `assessment.json`, `result.json`, the immutable run source and PnL. A
   completed process is not a passing strategy. Read every blocking result.
6. Append the returned outcome to the study ledger. `results.tsv` records source
   hash, net Sharpe, status and description. Blank metrics denote failed runs;
   never turn a crash into a numerical zero. Do not commit study outputs.
7. `keep` means the frozen development screen passed; `discard` retains the
   source and evidence of a failed screen. A discarded candidate can inform a
   new, separately recorded experiment. Neither result is final confirmation.
8. Continue from the recorded hypothesis. Preserve parent sources and the full
   history; do not reset another worktree or erase failed results.

Each evaluation is bounded to 20 minutes and the registry records incomplete
attempts. A repeated infrastructure failure requires diagnosis before another
retry. Harness² owns Propose–Probe–Compose, adoption and the persistent library;
its adoption selects a working harness, not an approved investment strategy.

Use arithmetic annualization, fixed-book PnL and the exact evaluator's units.
Keep signal IC, portfolio PnL, reproducibility and final confirmation distinct.
Repeated development search is adaptive. A DSR screen or a winner-only bootstrap
does not certify selection-adjusted significance. Keep the final holdout closed.
