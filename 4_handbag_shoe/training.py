# Handbag-Shoe binary classifier
# Based on MIT 15.773 Hands-On Deep Learning
# Original colab: https://colab.research.google.com/drive/1q42h-IJTmxjKeG5f4E_2v7jQW0vDZi0T
#
# Dataset: Handbags vs Shoes (small custom dataset, 224x224 RGB)
# Models: cnn, cnn_augmented, cnn_residual, resnet50, resnet50_e2e
# Usage:
#   uv run python training.py [model]           # Keras (default)
#   uv run python training.py [model] --torch   # PyTorch

import sys
import time

from common import prepare_data, print_metrics, log_results_csv

# --- Parse flags ---
use_torch = "--torch" in sys.argv
args = [a for a in sys.argv[1:] if a != "--torch"]

# --- Configuration ---

MODEL = args[0] if args else "cnn"
NUM_EPOCHS = {"cnn": 20, "cnn_augmented": 40, "cnn_residual": 40, "resnet50": 20, "resnet50_e2e": 20}
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

VALID_MODELS = ["cnn", "cnn_augmented", "cnn_residual", "resnet50", "resnet50_e2e"]
if MODEL not in VALID_MODELS:
    print(f"Unknown model: {MODEL}. Choose from: {', '.join(VALID_MODELS)}")
    sys.exit(1)

num_epochs = NUM_EPOCHS[MODEL]

# --- Data ---

base_dir = prepare_data("./data")

# --- Framework dispatch ---

t0 = time.time()

if use_torch:
    import torch
    torch.manual_seed(42)
    from common import load_datasets_torch, load_datasets_torch_resnet
    from model_cnn_torch import CNN, CNNResidual
    from model_resnet50_torch import get_resnet50_base, extract_features, ResNet50Head, ResNet50E2E
    from train_torch import train_model, evaluate_model

    if MODEL == "cnn":
        train_loader, val_loader, test_loader = load_datasets_torch(base_dir, batch_size=BATCH_SIZE)
        model = CNN()
        print(model)
        num_params = sum(p.numel() for p in model.parameters())
        history = train_model(model, train_loader, val_loader,
                              num_epochs=num_epochs, learning_rate=LEARNING_RATE,
                              data_mode="loader")
        training_seconds = round(time.time() - t0, 1)
        test_loss, test_accuracy = evaluate_model(model, test_loader, data_mode="loader")

    elif MODEL == "cnn_augmented":
        train_loader, val_loader, test_loader = load_datasets_torch(base_dir, batch_size=BATCH_SIZE, augment=True)
        model = CNN()
        print(model)
        num_params = sum(p.numel() for p in model.parameters())
        history = train_model(model, train_loader, val_loader,
                              num_epochs=num_epochs, learning_rate=LEARNING_RATE,
                              data_mode="loader")
        training_seconds = round(time.time() - t0, 1)
        test_loss, test_accuracy = evaluate_model(model, test_loader, data_mode="loader")

    elif MODEL == "cnn_residual":
        train_loader, val_loader, test_loader = load_datasets_torch(base_dir, batch_size=BATCH_SIZE)
        model = CNNResidual()
        print(model)
        num_params = sum(p.numel() for p in model.parameters())
        history = train_model(model, train_loader, val_loader,
                              num_epochs=num_epochs, learning_rate=LEARNING_RATE,
                              data_mode="loader")
        training_seconds = round(time.time() - t0, 1)
        test_loss, test_accuracy = evaluate_model(model, test_loader, data_mode="loader")

    elif MODEL == "resnet50":
        train_loader, val_loader, test_loader = load_datasets_torch_resnet(base_dir, batch_size=BATCH_SIZE)
        resnet50_base = get_resnet50_base()

        print("Extracting features with ResNet50...")
        train_features, train_labels = extract_features(train_loader, resnet50_base)
        val_features, val_labels = extract_features(val_loader, resnet50_base)
        test_features, test_labels = extract_features(test_loader, resnet50_base)

        model = ResNet50Head()
        print(model)
        num_params = sum(p.numel() for p in model.parameters())
        history = train_model(model, (train_features, train_labels), (val_features, val_labels),
                              num_epochs=num_epochs, batch_size=BATCH_SIZE,
                              learning_rate=LEARNING_RATE, data_mode="numpy")
        training_seconds = round(time.time() - t0, 1)
        test_loss, test_accuracy = evaluate_model(model, (test_features, test_labels), data_mode="numpy")

    elif MODEL == "resnet50_e2e":
        train_loader, val_loader, test_loader = load_datasets_torch_resnet(base_dir, batch_size=BATCH_SIZE, augment=True)
        model = ResNet50E2E()
        print(model)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        history = train_model(model, train_loader, val_loader,
                              num_epochs=num_epochs, learning_rate=LEARNING_RATE,
                              data_mode="loader")
        training_seconds = round(time.time() - t0, 1)
        test_loss, test_accuracy = evaluate_model(model, test_loader, data_mode="loader")

else:
    import keras
    keras.utils.set_random_seed(42)
    from common import load_datasets
    from model_cnn import build_cnn, build_cnn_augmented, build_cnn_residual
    from model_resnet50 import get_resnet50_base, extract_features, build_resnet50_head, build_resnet50_e2e

    train_dataset, validation_dataset, test_dataset = load_datasets(base_dir, batch_size=BATCH_SIZE)

    if MODEL == "cnn":
        model = build_cnn()
    elif MODEL == "cnn_augmented":
        model = build_cnn_augmented()
    elif MODEL == "cnn_residual":
        model = build_cnn_residual()
    elif MODEL == "resnet50":
        resnet50_base = get_resnet50_base()
        print("Extracting features with ResNet50...")
        train_features, train_labels = extract_features(train_dataset, resnet50_base)
        val_features, val_labels = extract_features(validation_dataset, resnet50_base)
        test_features, test_labels = extract_features(test_dataset, resnet50_base)
        model = build_resnet50_head()
    elif MODEL == "resnet50_e2e":
        model = build_resnet50_e2e()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    if MODEL == "resnet50":
        history_obj = model.fit(
            train_features, train_labels,
            epochs=num_epochs,
            validation_data=(val_features, val_labels),
        )
    else:
        history_obj = model.fit(
            train_dataset, epochs=num_epochs, validation_data=validation_dataset
        )

    training_seconds = round(time.time() - t0, 1)
    history = history_obj.history
    num_params = model.count_params()

    if MODEL == "resnet50":
        test_loss, test_accuracy = model.evaluate(test_features, test_labels)
    else:
        test_loss, test_accuracy = model.evaluate(test_dataset)

# --- Results (shared) ---

train_loss = history["loss"][-1]
train_accuracy = history["accuracy"][-1]
val_loss = history["val_loss"][-1]
val_accuracy = history["val_accuracy"][-1]

print_metrics(test_accuracy, test_loss, val_accuracy, val_loss,
              train_accuracy, train_loss, num_params, num_epochs,
              training_seconds)

log_results_csv(test_accuracy, test_loss, val_accuracy, val_loss,
                num_params, training_seconds,
                framework="torch" if use_torch else "keras", model=MODEL)
