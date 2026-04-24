# Autoresearch: Autonomous Experiment Protocol

Run autonomous deep learning experiments: `$ARGUMENTS`

Arguments: `<folder>` or `<folder> <model>`
- Single-model projects: `/autoresearch 1_heart-disease`
- Multi-model projects: `/autoresearch 4_handbag_shoe cnn`

## Setup

1. Parse `$ARGUMENTS`: first word is the folder, second word (if present) is the model name.
2. `cd` into the folder.
3. Read `CLAUDE.md` and `training.py` in that folder for full context.
4. If a model name was given, also read the model file specified in `CLAUDE.md` (e.g. `model_cnn.py`).
5. Read `results.csv` if it exists — this is your experiment history.
6. Confirm setup looks good, then begin.

## Modifiable files

- **Single-model** (no model arg): only `training.py`.
- **Multi-model** (model arg given): `training.py` (hyperparams: LR, epochs, batch size, optimizer, loss) AND the model's builder file (architecture: layers, widths, depths, regularization). The folder's `CLAUDE.md` maps model names to files. Do NOT modify `common.py` or other model files.

## Rules

**You CAN:** modify hyperparameters, optimizer settings, regularization, preprocessing, and architecture tweaks (add/remove layers, change widths, add dropout/batchnorm, change activations).

**You CANNOT:**
- Install new packages or add dependencies beyond `pyproject.toml`
- Change the dataset source or train/test split
- Change the random seed
- Modify the evaluation (test set metrics must remain honest)
- Switch to a fundamentally different architecture (e.g. don't turn a CNN into a ResNet)

**Goal: maximize test_accuracy.** Simpler is better — don't add complexity for marginal gains.

## Continuation vs fresh start

- If `results.csv` **exists and has data**: this is a continuation. Review the history to understand what's been tried, what the current best is, and what ideas failed. Do NOT re-run the baseline. Pick up where the last session left off — propose new ideas informed by past results.
- If `results.csv` **doesn't exist or is empty**: this is a fresh start. Run `training.py` as-is to establish the baseline first.

## Experiment loop

LOOP FOREVER:

1. Edit the modifiable file(s) with an experimental idea.
2. Stage and commit:
   - Single-model: `git add training.py && git commit -m "description"`
   - Multi-model: `git add training.py model_*.py && git commit -m "description"`
3. Run training:
   - Single-model: `uv run python training.py > run.log 2>&1`
   - Multi-model: `uv run python training.py <model> > run.log 2>&1`
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
