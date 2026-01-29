import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from . import config
from .dataset import load_csv

def main():
    model = tf.keras.models.load_model(os.path.join(config.RESULTS_DIR, "model.keras"))
    X, y = load_csv(config.DATA_PATH)
    preds = model.predict(X).reshape(-1)

    os.makedirs(os.path.join(config.RESULTS_DIR, "metrics"), exist_ok=True)

    if config.TASK == "classification":
        yhat = (preds >= 0.5).astype(int)
        acc = accuracy_score(y.astype(int), yhat)
        f1 = f1_score(y.astype(int), yhat)
        print("Accuracy:", acc)
        print("F1:", f1)
    else:
        mae = mean_absolute_error(y, preds)
        mse = mean_squared_error(y, preds)
        print("MAE:", mae)
        print("MSE:", mse)

if __name__ == "__main__":
    main()

