import os, json
from tensorflow import keras
from . import config
from .utils import set_seed
from .dataset import load_csv, split_and_scale
from .model import build_mlp

def main():
    set_seed(config.RANDOM_SEED)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "metrics"), exist_ok=True)

    X, y = load_csv(config.DATA_PATH)
    X_train, X_val, y_train, y_val, scaler = split_and_scale(
        X, y, config.VAL_SPLIT, config.RANDOM_SEED
    )

    model = build_mlp(
        input_dim=X_train.shape[1],
        hidden=config.HIDDEN,
        dropout=config.DROPOUT,
        task=config.TASK
    )

    if config.TASK == "classification":
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        loss = "mse"
        metrics = ["mae"]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss,
        metrics=metrics,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    model.save(os.path.join(config.RESULTS_DIR, "model.keras"))

    with open(os.path.join(config.RESULTS_DIR, "metrics", "history.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    print("Saved model + history in results/")

if __name__ == "__main__":
    main()


