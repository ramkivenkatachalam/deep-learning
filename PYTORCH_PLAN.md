# Plan: Add PyTorch Support Across All Projects

Add `--torch` flag support to each project folder, following the pattern established in `1_heart-disease/`.

## Completed

### 1_heart-disease (done)
- Created `common.py`, `model_keras.py`, `model_torch.py`, `train_torch.py`
- Refactored `training.py` with `--torch` dispatch
- Created `course_notebook_torch.ipynb`
- Results: Keras 93.44%, PyTorch 91.80%

## Remaining Projects

### 2_fashion-mnist
- **Type**: Single-model (MLP), inline architecture in `training.py`
- **Architecture**: Flatten → Dense(1024, gelu) → BN → Dropout(0.4) → Dense(512) → ... → Dense(10, softmax)
- **Special**: CosineDecay LR schedule, label smoothing, AdamW
- **Work needed**:
  1. Create `common.py` — extract data loading (Fashion-MNIST via keras.datasets), metrics, CSV logging
  2. Create `model_keras.py` — extract the Sequential model definition
  3. Create `model_torch.py` — `nn.Module` with same architecture (Linear + GELU + BatchNorm1d + Dropout)
  4. Create `train_torch.py` — training loop with CosineAnnealingLR, CrossEntropyLoss
  5. Refactor `training.py` — `--torch` dispatch, shared data loading
  6. Create `course_notebook_torch.ipynb` from `course_notebook.ipynb`
  7. Update `CLAUDE.md` and `README.md`
- **Notes**: No `common.py` exists yet — everything is inline. Label smoothing needs manual implementation in PyTorch (or use `CrossEntropyLoss(label_smoothing=0.1)`).

### 3_fashion_mnist_cnn
- **Type**: Single-model (CNN), inline architecture in `training.py`
- **Architecture**: 3x(Conv2D → BN → Conv2D → BN → MaxPool → Dropout) → Flatten → Dense(512) → Dense(10, softmax)
- **Special**: tf.data augmentation pipeline (random flip, reflect pad + random crop), CosineDecay LR
- **Work needed**:
  1. Create `common.py` — extract data loading, augmentation (torchvision.transforms), metrics, CSV logging
  2. Create `model_keras.py` — extract model definition
  3. Create `model_torch.py` — `nn.Module` with Conv2d + BatchNorm2d + ReLU + MaxPool2d + Dropout
  4. Create `train_torch.py` — training loop with CosineAnnealingLR
  5. Refactor `training.py` — `--torch` dispatch
  6. No course notebook exists for this project
  7. Update `CLAUDE.md`
- **Notes**: Data augmentation is the trickiest part — need to replicate `tf.data` pipeline using `torchvision.transforms` (RandomHorizontalFlip, Pad + RandomCrop). Channel ordering differs (Keras: NHWC, PyTorch: NCHW).

### 4_handbag_shoe
- **Type**: Multi-model (`cnn`, `cnn_augmented`, `cnn_residual`, `resnet50`, `resnet50_e2e`)
- **Architecture**: Various CNNs + ResNet50 transfer learning
- **Special**: `common.py` already exists, image datasets loaded from directories, ResNet50 feature caching
- **Work needed**:
  1. Update `common.py` — add PyTorch data loading (torchvision.datasets.ImageFolder + DataLoader)
  2. Create `model_cnn_torch.py` — PyTorch versions of CNN, CNN+augmentation, CNN+residual
  3. Create `model_resnet50_torch.py` — PyTorch ResNet50 (torchvision.models), feature extraction + end-to-end
  4. Create `train_torch.py` — training loop for both CNN and transfer learning variants
  5. Update `training.py` — `--torch` dispatch for each model variant
  6. Create PyTorch course notebooks (3 Keras notebooks exist)
  7. Update `CLAUDE.md`
- **Notes**: Most complex project. ResNet50 feature caching needs reworking for PyTorch. Data augmentation via `torchvision.transforms`. Image loading with `ImageFolder` instead of `keras.utils.image_dataset_from_directory`. Consider doing CNN models first, then transfer learning separately.

### 5_music_genre_classification
- **Type**: Multi-model (`unigram`, `bigram`)
- **Architecture**: Dense classifiers on bag-of-words (multi-hot) input
- **Special**: `common.py` exists, uses Keras `TextVectorization` layer for tokenization
- **Work needed**:
  1. Update `common.py` — add PyTorch-compatible text vectorization (scikit-learn CountVectorizer or manual vocab building)
  2. Create `model_dense_torch.py` — simple Dense models for unigram/bigram
  3. Create `train_torch.py` — training loop with CrossEntropyLoss
  4. Update `training.py` — `--torch` dispatch
  5. Create `course_notebook_torch.ipynb` from `course_notebook.ipynb`
  6. Update `CLAUDE.md`
- **Notes**: The tricky part is replacing `TextVectorization` — this is a Keras preprocessing layer. Options: (a) use scikit-learn's `CountVectorizer` in common.py, (b) build vocab manually with collections.Counter. The model architectures themselves are trivial Dense networks.

### 6_standalone_word_embeddings
- **Type**: Multi-model (`glove_frozen`, `glove_finetune`, `custom`)
- **Architecture**: Embedding → Pooling → Dense classifier
- **Special**: `common.py` exists with GloVe loading, uses `TextVectorization` (int mode) + `Embedding` layer
- **Work needed**:
  1. Update `common.py` — add PyTorch-compatible tokenization (build vocab, convert to integer sequences)
  2. Create `model_embedding_torch.py` — `nn.Embedding` with GloVe init (frozen/trainable), custom random init
  3. Create `train_torch.py` — training loop with CrossEntropyLoss, padding/collation for variable-length sequences
  4. Update `training.py` — `--torch` dispatch
  5. Create PyTorch notebooks (2 Keras notebooks exist)
  6. Update `CLAUDE.md`
- **Notes**: Need to handle GloVe embedding matrix initialization for `nn.Embedding`. Frozen embeddings: `embedding.weight.requires_grad = False`. Variable-length sequences need padding in DataLoader collate_fn or pre-padding. `TextVectorization` replacement is shared concern with project 5 — consider a shared utility or solve in project 5 first.

## Suggested Order

1. **2_fashion-mnist** — simplest (Dense-only MLP, no text/images complexities)
2. **3_fashion_mnist_cnn** — adds Conv2d and data augmentation
3. **5_music_genre_classification** — introduces text vectorization challenge
4. **6_standalone_word_embeddings** — builds on text vectorization + adds embeddings
5. **4_handbag_shoe** — most complex (multiple CNN variants + transfer learning)

## Common Patterns (from 1_heart-disease)

- `model_torch.py` contains only architecture (`nn.Module` subclass + `build()` function)
- `train_torch.py` contains `train_model()` and `evaluate_model()` functions
- `training.py` checks `"--torch" in sys.argv` and lazy-imports the appropriate framework
- `common.py` is framework-agnostic (numpy/pandas only)
- Per-framework learning rates when optimal values differ
- `results.csv` has `framework` column to distinguish Keras vs PyTorch experiments
- `plot_results.py` generates separate charts per framework automatically
