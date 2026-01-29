import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_csv(path: str):
    df = pd.read_csv(path)
    if "y" not in df.columns:
        raise ValueError("CSV must contain a target column named 'y'.")
    y = df["y"].to_numpy().astype(np.float32)
    X = df.drop(columns=["y"]).to_numpy().astype(np.float32)
    return X, y

def split_and_scale(X, y, val_split: float, seed: int):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train, X_val, y_train, y_val, scaler

