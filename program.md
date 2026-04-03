# Autonomous Experiment Protocol (Autoresearch)

This protocol drives autonomous deep learning experiments. Each assignment lives in its own folder (e.g. `1_heart-disease/`, `2_fashion-mnist/`) with a `training.py` that you iteratively improve.

## Setup

To set up a new experiment run, work with the user to:

1. **`cd` into the assignment folder** (e.g. `cd 1_heart-disease`).
2. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr02`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
3. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
4. **Read the in-scope files**: Read the assignment's `CLAUDE.md` and `training.py` for full context.
5. **Initialize results.csv**: Create `results.csv` with just the header row if it doesn't exist. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

You launch training simply as: `uv run python training.py`.

**What you CAN do:**
- Modify `training.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, regularization, preprocessing, feature engineering, etc.

**What you CANNOT do:**
- Install new packages or add dependencies beyond what's already in `pyproject.toml`.
- Change the dataset source or the train/test split.
- Change the random seed (41) — results must be reproducible.
- Modify the evaluation (test set metrics must remain honest).

**The goal is simple: get the highest test_accuracy.** Everything else is fair game. The only constraint is that the code runs without crashing.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
test_accuracy:    0.9016
test_loss:        0.3204
val_accuracy:     0.7959
val_loss:         0.5150
train_accuracy:   0.9275
train_loss:       0.1684
num_params:       249
num_epochs:       300
training_seconds: 5.5
```

You can extract the key metric from the log file:

```
grep "^test_accuracy:" run.log
```

## Logging results

Results are auto-appended to `results.csv` after each run. The CSV has these columns:

```
commit,test_accuracy,test_loss,val_accuracy,val_loss,num_params,training_seconds,status,description
```

1. `commit` — git short hash (7 chars), or "uncommitted" if not committed yet
2. `test_accuracy` — primary metric (higher is better)
3. `test_loss` — loss on test set
4. `val_accuracy` — last epoch validation accuracy
5. `val_loss` — last epoch validation loss
6. `num_params` — total model parameters
7. `training_seconds` — wall clock training time
8. `status` — `pending` (auto-logged), then updated to `keep`, `discard`, or `crash`
9. `description` — short text description of what this experiment tried

NOTE: do not commit `results.csv` — leave it untracked by git (it's in `.gitignore`).

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr02`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `training.py` with an experimental idea by directly hacking the code.
3. git commit the change.
4. Run the experiment: `uv run python training.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
5. Read out the results: `grep "^test_accuracy:\|^test_loss:\|^val_accuracy:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
7. Update the status and description for this run's row in `results.csv`.
8. If test_accuracy improved (higher), you "advance" the branch, keeping the git commit.
9. If test_accuracy is equal or worse, you git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take under 2 minutes. If a run exceeds 5 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the CSV, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical architectural changes, revisit ideas with different hyperparameters. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each experiment takes ~30 seconds to a couple of minutes, so you can run many experiments per hour. The user then wakes up to experimental results, all completed by you while they slept!
