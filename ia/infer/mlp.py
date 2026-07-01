"""
IA/infer/mlp.py — Inférence des modèles MLP (rétro-compatibilité V2).

Fonctions :
  - load_mlp(path)      : charge un modèle MLP depuis un fichier pickle (.gy)
  - predict_mlp(model, X): effectue une prédiction avec le modèle MLP chargé

Usage :
    from IA import load_mlp, predict_mlp
    model = load_mlp("mini_mlp.gy")
    result = predict_mlp(model, X_new)
"""

import pickle
import logging

import numpy as np

logger = logging.getLogger(__name__)


def load_mlp(path):
    """
    Charge un modèle MLP depuis un fichier pickle (.gy).

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du modèle (weights, biases, config).
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("MLP charge depuis %s", path)
    return model


def predict_mlp(model, X):
    """
    Prédiction avec un modèle MLP chargé.

    Effectue un forward pass complet à travers toutes les couches :
      Dense -> ReLU (couches cachées) -> sigmoid/softmax (couche de sortie).

    Args:
        model: Dictionnaire de paramètres (issu de load_mlp).
        X: array-like d'entrée, shape (n_samples, n_features) ou (n_features,).

    Returns:
        dict: {
            'output': ndarray,
            'predictions': ndarray (classes prédites),
            'probabilities': ndarray (si multiclass),
        }
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    weights = model['weights']
    biases = model['biases']
    multiclass = model.get('multiclass', False)
    n_layers = len(weights)
    unique_classes = model.get('unique_classes', [0, 1])

    # Forward pass
    a = X
    for i in range(n_layers):
        z = a @ weights[i] + biases[i]
        if i < n_layers - 1:
            a = np.maximum(0.0, z)  # ReLU
        else:
            if multiclass:
                e = np.exp(z - z.max(axis=1, keepdims=True))
                a = e / e.sum(axis=1, keepdims=True)  # softmax
            else:
                a = 1.0 / (1.0 + np.exp(-z))  # sigmoid

    # Résultats
    if multiclass:
        predictions = a.argmax(axis=1)
        # Remapper les indices aux classes originales
        class_preds = np.array([unique_classes[p] for p in predictions])
        return {
            'output': a,
            'predictions': class_preds,
            'probabilities': a,
        }
    else:
        predictions = (a > 0.5).astype(int).flatten()
        return {
            'output': a,
            'predictions': predictions,
        }