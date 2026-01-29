from tensorflow import keras
from tensorflow.keras import layers

def build_mlp(input_dim: int, hidden: list[int], dropout: float, task: str):
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for h in hidden:
        x = layers.Dense(h, activation="relu")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)

    if task == "classification":
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(1)(x)

    return keras.Model(inputs, outputs)

