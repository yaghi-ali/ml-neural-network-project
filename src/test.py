import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tensorflow.keras import models, layers, optimizers, callbacks


# 1) Chargement automatique de tous les fichiers .pickle


DATAPATH = r"C:/Users/manar/Downloads/s20251110_all_donnees/s20251110/"
flist = sorted(glob.glob(os.path.join(DATAPATH, "s20251110_*.pickle")))
print("Nombre de fichiers trouvés :", len(flist))

def read_pickle(path):
    with open(path, "rb") as fd:
        return pickle.load(fd)


# 2) Préparation des données :


DOWNSAMPLE = 500       
WIN = 256              # longueur fenêtre (à ajuster)
STRIDE = 64            # pas de glissement (overlap) : WIN=256, stride=64 => bon compromis
POS_SG_WIN = 105       # Savitzky-Golay
POS_SG_POLY = 5

TARGET_MODE = "sequence"   # "sequence" ou "center"

def sliding_windows(x, win, stride):
    # retourne un tableau (nwin, win)
    n = 1 + (len(x) - win) // stride
    out = np.zeros((n, win), dtype=np.float32)
    for i in range(n):
        a = i * stride
        out[i] = x[a:a+win]
    return out

def makedata_position(d):
    # Bruts
    sig_raw = np.asarray(d["data"][0], dtype=np.float32)   # channel interférences
    pos_raw = np.asarray(d["data"][1], dtype=np.float32)   # channel position (déplacement imposé)

    # Sous-échantillonnage identique sur les 2 voies
    sig = sig_raw[::DOWNSAMPLE]
    pos = pos_raw[::DOWNSAMPLE]

    # Lissage position (optionnel mais utile si bruit capteur)
    if len(pos) >= POS_SG_WIN:
        pos = signal.savgol_filter(pos, POS_SG_WIN, POS_SG_POLY).astype(np.float32)

    # --- Fenêtrage glissant ---
    # (nwin, WIN)
    X = sliding_windows(sig, WIN, STRIDE)
    Y = sliding_windows(pos, WIN, STRIDE)

    if TARGET_MODE == "center":
        # y scalaire = position au centre de la fenêtre
        y = Y[:, WIN // 2]
        return X, y
    else:
        # y séquence = position sur toute la fenêtre
        return X, Y


# AJOUT : figures signal réel + cible (position) sur un fichier


def plot_raw_and_target_one_file(d, title_prefix=""):
    sig_raw = np.asarray(d["data"][0], dtype=np.float32)
    pos_raw = np.asarray(d["data"][1], dtype=np.float32)

    # mêmes opérations que dans makedata_position
    sig = sig_raw[::DOWNSAMPLE]
    pos = pos_raw[::DOWNSAMPLE]

    pos_s = pos.copy()
    if len(pos_s) >= POS_SG_WIN:
        pos_s = signal.savgol_filter(pos_s, POS_SG_WIN, POS_SG_POLY).astype(np.float32)

    # Figure 1 : signal d'interférences
    plt.figure(figsize=(12,4))
    plt.plot(sig_raw, label="Signal brut")
    plt.plot(np.arange(0, len(sig_raw), DOWNSAMPLE), sig, label=f"Signal downsample (/{DOWNSAMPLE})")
    plt.title(f"{title_prefix}Signal d'interférences (réel)")
    plt.xlabel("Indice échantillon")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Figure 2 : cible (position)
    plt.figure(figsize=(12,4))
    plt.plot(pos_raw, label="Position brute")
    plt.plot(np.arange(0, len(pos_raw), DOWNSAMPLE), pos, label=f"Position downsample (/{DOWNSAMPLE})")
    plt.plot(np.arange(0, len(pos_raw), DOWNSAMPLE), pos_s, label="Position lissée (Savitzky-Golay)")
    plt.title(f"{title_prefix}Cible (déplacement/position) — réelle")
    plt.xlabel("Indice échantillon")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# AJOUT : afficher les signaux (1er fichier)


if len(flist) > 0:
    d0 = read_pickle(flist[0])
    plot_raw_and_target_one_file(d0, title_prefix="Exemple fichier 0 — ")


# 3) Construire dataset complet + split par fichiers


# Split fichiers : 80/20
rng = np.random.default_rng(0)
perm = rng.permutation(len(flist))
ntrain_files = int(0.8 * len(flist))
train_files = [flist[i] for i in perm[:ntrain_files]]
val_files   = [flist[i] for i in perm[ntrain_files:]]

def build_dataset(file_list):
    X_all, Y_all = [], []
    for f in file_list:
        d = read_pickle(f)
        X, Y = makedata_position(d)
        X_all.append(X)
        Y_all.append(Y)
    X_all = np.vstack(X_all)
    Y_all = np.vstack(Y_all) if TARGET_MODE == "sequence" else np.hstack(Y_all)
    return X_all, Y_all

X_train, y_train = build_dataset(train_files)
X_val,   y_val   = build_dataset(val_files)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape,   "y_val  :", y_val.shape)


# AJOUT : exemple fenêtre X vs fenêtre Y


k = 0
plt.figure(figsize=(12,4))
plt.plot(X_train[k], label="X fenêtre (signal interférences)")
plt.title("Exemple fenêtre d'entrée X (signal)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

if TARGET_MODE == "sequence":
    plt.figure(figsize=(12,4))
    plt.plot(y_train[k], label="Y fenêtre (position cible)")
    plt.title("Exemple fenêtre cible Y (position)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Exemple cible (position centre) y_train[k] =", y_train[k])


# 4) Normalisation (TRÈS important)


# Normalisation X : mean/std calculés sur train seulement
X_mean = X_train.mean()
X_std  = X_train.std() + 1e-8
X_train_n = (X_train - X_mean) / X_std
X_val_n   = (X_val   - X_mean) / X_std

if TARGET_MODE == "sequence":
    y_mean = y_train.mean()
    y_std  = y_train.std() + 1e-8
    y_train_n = (y_train - y_mean) / y_std
    y_val_n   = (y_val   - y_mean) / y_std
else:
    y_mean = y_train.mean()
    y_std  = y_train.std() + 1e-8
    y_train_n = (y_train - y_mean) / y_std
    y_val_n   = (y_val   - y_mean) / y_std

# reshape pour Conv1D / LSTM : (batch, time, channels)
X_train_n = X_train_n[..., None]
X_val_n   = X_val_n[..., None]

if TARGET_MODE == "sequence":
    # y doit être (batch, time, 1) pour sortie séquence
    y_train_n = y_train_n[..., None]
    y_val_n   = y_val_n[..., None]

print("Après reshape:")
print("X_train_n:", X_train_n.shape, "y_train_n:", y_train_n.shape)


# 5) Modèle : CNN


def build_model_sequence(win):
    inp = layers.Input(shape=(win, 1))

    x = layers.Conv1D(16, 7, padding="same", activation="relu")(inp)
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)  # win/2

    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)  # win/4

    # Remonter à la résolution temporelle initiale
    x = layers.UpSampling1D(2)(x)  # win/2
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)

    x = layers.UpSampling1D(2)(x)  # win
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)

    # LSTM bidirectionnel pour capturer la phase / dynamique
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)

    # Sortie : position normalisée (sequence)
    out = layers.Conv1D(1, 1, padding="same", activation="linear")(x)

    return models.Model(inp, out)

def build_model_center(win):
    inp = layers.Input(shape=(win, 1))

    x = layers.Conv1D(16, 7, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)

    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)

    return models.Model(inp, out)

if TARGET_MODE == "sequence":
    M = build_model_sequence(WIN)
else:
    M = build_model_center(WIN)

M.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="mse"
)

M.summary()


# 6) Entraînement (avec early stopping + reduce LR)


cbs = [
    callbacks.EarlyStopping(patience=30, restore_best_weights=True, monitor="val_loss"),
    callbacks.ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6, monitor="val_loss"),
]

history = M.fit(
    X_train_n, y_train_n,
    validation_data=(X_val_n, y_val_n),
    epochs=1000,
    batch_size=64,
    verbose=1,
    callbacks=cbs
)


# 7) Courbes de loss


plt.figure()
plt.semilogy(history.history["loss"], label="Train loss")
plt.semilogy(history.history["val_loss"], label="Val loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MSE (log)")
plt.grid(True)
plt.show()


# 8) Prédiction + dénormalisation + visualisation


pred_n = M.predict(X_val_n)

# Dénormaliser
pred = pred_n * y_std + y_mean
true = y_val_n * y_std + y_mean

# Affichage
if TARGET_MODE == "sequence":
    k = 0  # fenêtre à afficher
    plt.figure(figsize=(10,5))
    plt.plot(true[k,:,0], label="True position (window)")
    plt.plot(pred[k,:,0], label="Pred position (window)")
    plt.legend()
    plt.title("Fenêtre: position vraie vs prédite")
    plt.grid(True)
    plt.show()
else:
    plt.figure(figsize=(10,5))
    plt.plot(true[:200], label="True position (center)")
    plt.plot(pred[:200], label="Pred position (center)")
    plt.legend()
    plt.title("Position au centre: vraie vs prédite")
    plt.grid(True)
    plt.show()


# 9) Reconstruction d'une trajectoire complète (mode sequence)


def reconstruct_from_windows(windows_pred, win, stride):
    # windows_pred: (nwin, win)
    nwin = windows_pred.shape[0]
    L = (nwin - 1) * stride + win
    y = np.zeros(L, dtype=np.float32)
    w = np.zeros(L, dtype=np.float32)

    for i in range(nwin):
        a = i * stride
        y[a:a+win] += windows_pred[i]
        w[a:a+win] += 1.0

    return y / np.maximum(w, 1e-8)

if TARGET_MODE == "sequence":
    recon_pred = reconstruct_from_windows(pred[:, :, 0], WIN, STRIDE)
    recon_true = reconstruct_from_windows(true[:, :, 0], WIN, STRIDE)

    plt.figure(figsize=(12,5))
    plt.plot(recon_true[:2000], label="True reconstructed")
    plt.plot(recon_pred[:2000], label="Pred reconstructed")
    plt.legend()
    plt.title("Reconstruction (overlap-add) - aperçu")
    plt.grid(True)
    plt.show()


# 10) Parity plot : signal réel (X) vs signal prédit (Y)


k = 0  

x_real = true[k, :, 0]
y_pred = pred[k, :, 0]

plt.figure(figsize=(6,6))
plt.plot(x_real, y_pred, ".", alpha=0.6, label="Prédictions")

# droite idéale y = x
m = min(x_real.min(), y_pred.min())
M = max(x_real.max(), y_pred.max())
plt.plot([m, M], [m, M], "r--", label="y = x")

plt.xlabel("déplacement réel")
plt.ylabel("déplacement interférer")
plt.title("réel (X) vs interférer (Y)")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()

