# Deep Learning Class Assignments

Each assignment lives in a numbered folder with a `training.py` file:
- `1_heart-disease/` — UCI Heart Disease binary classifier
- `2_fashion-mnist/` — Fashion-MNIST 10-class image classifier
- `4_handbag_shoe/` — Handbag vs Shoe binary classifier (CNN + ResNet50 transfer learning)
- `5_music_genre_classification/` — Music genre from lyrics, 3-class classifier (dense BoW models)
- `6_standalone_word_embeddings/` — Music genre from lyrics, 3-class classifier (word embedding models)

## PyTorch support
Add `--torch` to run with PyTorch instead of Keras (default):
```
python training.py --torch              # single-model folders
python training.py cnn --torch          # multi-model folders
```

## Autoresearch Protocol

Run `/autoresearch <folder> [model]` to start autonomous experiments.
- Single-model: `/autoresearch 1_heart-disease`
- Multi-model: `/autoresearch 4_handbag_shoe cnn`

Key points:
- `cd` into the assignment folder before starting
- Single-model projects: only modify `training.py`
- Multi-model projects: modify `training.py` (hyperparams) + the model's builder file (architecture). See the folder's `CLAUDE.md` for model→file mapping.
- Run with `uv run python training.py [model]`
- Use `../run_experiment.sh "description" [model]` to commit + run + show results

## Shared files (root level)
- `run_experiment.sh` — commit + run helper
- `plot_results.py` — visualize results.csv from any assignment folder
- `pyproject.toml` — shared Python dependencies
