# Autoresearch: Autonomous Experiment Protocol

Run autonomous deep learning experiments on the assignment folder: `$ARGUMENTS`

## Setup

1. `cd` into the `$ARGUMENTS` folder.
2. Read `CLAUDE.md` and `training.py` in that folder for full context.
3. If `results.csv` doesn't exist, it will be created automatically on first run.
4. Confirm setup looks good, then begin.

## Rules

**You CAN:** modify `training.py` — architecture, optimizer, hyperparameters, training loop, batch size, regularization, preprocessing, feature engineering. Everything is fair game.

**You CANNOT:**
- Install new packages or add dependencies beyond `pyproject.toml`
- Change the dataset source or train/test split
- Change the random seed (41)
- Modify the evaluation (test set metrics must remain honest)

**Goal: maximize test_accuracy.** Simpler is better — don't add complexity for marginal gains.

**First run:** always establish the baseline by running `training.py` as-is.

## Experiment loop

LOOP FOREVER:

1. Edit `training.py` with an experimental idea.
2. `git add training.py && git commit -m "description of change"`
3. `uv run python training.py > run.log 2>&1`
4. `grep "^test_accuracy:\|^test_loss:\|^val_accuracy:" run.log`
5. If grep is empty → crashed. `tail -n 50 run.log` to diagnose. Fix if trivial, skip if broken.
6. Update status and description in `results.csv` for this run's row.
7. If test_accuracy **improved** → keep the commit.
8. If test_accuracy **equal or worse** → `git reset --hard HEAD~1` to revert.

## Important

- **NEVER STOP.** Do not ask "should I keep going?" — run until manually interrupted.
- **Timeout:** each run should be under 2 minutes. Kill and discard if over 5 minutes.
- **Crashes:** fix trivial bugs and re-run. Skip fundamentally broken ideas.
- **Redirect output:** always `> run.log 2>&1`. Never let training output flood context.
- **Do not commit results.csv** — it's gitignored.
