"""
IA/train/mlp.py — Entraînement MLP (Multi-Layer Perceptron).

Architecture : couches denses empilées avec ReLU + sortie sigmoid/softmax.
Conçu pour données tabulaires (classification et régression).

Usage démo :
    from IA.train.mlp import train_mlp
    result = train_mlp(epochs=100)

Usage réel :
    result = train_mlp(X=X, y=y, hidden_sizes=[64, 32], epochs=500, lr=0.01)
"""

import os
import logging

import numpy as np

from ..config import get_config, MODELS_DIR, MODEL_EXTENSION, ensure_directories
from ..ia_format import save_model, serialize_model_dict

logger = logging.getLogger(__name__)


# ============================================================================
# Activations
# ============================================================================

def relu(x):
    return np.maximum(0.0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float64)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1.0 - x)

def softmax(x):
    # x : (batch, n_classes)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ============================================================================
# Fonction principale
# ============================================================================

def train_mlp(X=None, y=None, hidden_sizes=None, lr=0.01, epochs=500,
              save_path=None, seed=42, batch_size=256, dropout=0.0):
    """
    Entraîne un MLP pour classification/régression tabulaire.

    Args:
        X: array (n_samples, n_features). None = données démo.
        y: array (n_samples, 1). None = données démo.
           Si les valeurs sont entières > 1, mode multi-classes activé.
        hidden_sizes: liste de tailles de couches cachées.
                      Défaut : [64, 32]
        lr: taux d'apprentissage.
        epochs: nombre d'époques.
        save_path: chemin de sauvegarde (.gy).
        seed: graine aléatoire.
        batch_size: taille de mini-batch (256 par défaut).
        dropout: taux de dropout entre couches cachées (0 = désactivé).

    Returns:
        dict : model, save_path, accuracy, history
    """
    ensure_directories()
    np.random.seed(seed)

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"mini_mlp{MODEL_EXTENSION}")

    if hidden_sizes is None:
        hidden_sizes = [64, 32]

    # --- Données démo --------------------------------------------------------
    if X is None or y is None:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((200, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float64).reshape(-1, 1)
        logger.info("Données démo générées : X=%s y=%s", X.shape, y.shape)

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_samples, n_features = X.shape

    # --- Détecter le mode ---------------------------------------------------
    unique_classes = np.unique(y).astype(int)
    n_classes = len(unique_classes)
    multiclass = n_classes > 2

    if multiclass:
        # One-hot encoding
        n_out = n_classes
        class_map = {v: i for i, v in enumerate(unique_classes)}
        y_enc = np.zeros((n_samples, n_out), dtype=np.float64)
        for i, val in enumerate(y.flatten()):
            y_enc[i, class_map[int(val)]] = 1.0
        logger.info("Mode multi-classes : %d classes", n_classes)
    else:
        n_out = 1
        y_enc = y
        logger.info("Mode binaire")

    # --- Initialisation des poids -------------------------------------------
    layer_sizes = [n_features] + hidden_sizes + [n_out]
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        # He initialization pour ReLU
        std = np.sqrt(2.0 / fan_in)
        W = np.random.randn(fan_in, fan_out) * std
        b = np.zeros((1, fan_out))
        weights.append(W)
        biases.append(b)

    n_layers = len(weights)
    history = []

    # --- Entraînement -------------------------------------------------------
    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(n_samples)
        X_shuf = X[idx]
        y_shuf = y_enc[idx]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]
            bs = len(Xb)

            # Forward pass
            activations = [Xb]
            masks = []  # stocker les masques de dropout par couche
            for i in range(n_layers):
                z = activations[-1] @ weights[i] + biases[i]
                if i < n_layers - 1:
                    a = relu(z)
                    # Dropout
                    if dropout > 0.0:
                        mask = (np.random.rand(*a.shape) > dropout).astype(np.float64)
                        masks.append(mask)
                        a = a * mask / (1.0 - dropout)
                    else:
                        masks.append(None)
                else:
                    masks.append(None)
                    # Couche de sortie
                    if multiclass:
                        a = softmax(z)
                    else:
                        a = sigmoid(z)
                activations.append(a)

            # Loss
            out = activations[-1]
            if multiclass:
                # Cross-entropy
                loss = -np.sum(yb * np.log(out + 1e-9)) / bs
            else:
                # BCE
                loss = -np.mean(yb * np.log(out + 1e-9) +
                                (1 - yb) * np.log(1 - out + 1e-9))
            epoch_loss += loss
            n_batches += 1

            # Backward pass
            if multiclass:
                delta = (out - yb) / bs
            else:
                delta = (out - yb) / bs

            grad_w = [None] * n_layers
            grad_b = [None] * n_layers

            for i in reversed(range(n_layers)):
                a_prev = activations[i]
                grad_w[i] = a_prev.T @ delta
                grad_b[i] = delta.sum(axis=0, keepdims=True)
                if i > 0:
                    delta = delta @ weights[i].T
                    # Propager a travers le dropout (inverted)
                    if dropout > 0.0 and masks[i - 1] is not None:
                        delta = delta * masks[i - 1] / (1.0 - dropout)
                    # Propager a travers ReLU
                    delta = delta * relu_deriv(activations[i])

            # Mise à jour
            for i in range(n_layers):
                weights[i] -= lr * grad_w[i]
                biases[i] -= lr * grad_b[i]

        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)

        if epoch % max(1, epochs // 10) == 0:
            logger.info("Epoch %d/%d  loss=%.6f", epoch, epochs, avg_loss)

    # --- Évaluation ---------------------------------------------------------
    # Forward complet sur tout X
    a = X
    for i in range(n_layers):
        z = a @ weights[i] + biases[i]
        if i < n_layers - 1:
            a = relu(z)
        else:
            a = softmax(z) if multiclass else sigmoid(z)

    if multiclass:
        preds = a.argmax(axis=1)
        trues = y.flatten().astype(int)
        # Remap classes
        trues_mapped = np.array([class_map[int(v)] for v in trues])
        accuracy = (preds == trues_mapped).mean() * 100.0
    else:
        preds = (a > 0.5).astype(int).flatten()
        trues = y.flatten().astype(int)
        accuracy = (preds == trues).mean() * 100.0

    logger.info("MLP entraîné — précision: %.2f%%", accuracy)

    # --- Sauvegarde ---------------------------------------------------------
    model = {
        'weights': weights,
        'biases': biases,
        'hidden_sizes': hidden_sizes,
        'n_features': n_features,
        'n_out': n_out,
        'n_classes': n_classes,
        'multiclass': multiclass,
        'unique_classes': unique_classes.tolist(),
        'layer_sizes': layer_sizes,
        'accuracy': accuracy,
    }

    header, tensors = serialize_model_dict(model)
    save_model(save_path, header, tensors)
    logger.info("MLP sauvegardé : %s (précision: %.1f%%)", save_path, accuracy)

    return {
        'model': model,
        'save_path': save_path,
        'accuracy': accuracy,
        'history': history,
    }
