"""
IA/infer/mlp.py — Inférence des modèles MLP.

Fonctions :
  - load_mlp(path)      : charge un modèle MLP depuis un fichier .gy
  - predict_mlp(model, X): effectue une prédiction avec le modèle MLP chargé

Usage :
    from IA import load_mlp, predict_mlp
    model = load_mlp("mini_mlp.gy")
    result = predict_mlp(model, X_new)
"""

import logging
from typing import Any, Dict

import numpy as np

from .._utils import merge_header, validate_model_path, validate_ndarray
from ..ia_format import load_model

logger = logging.getLogger(__name__)


def load_mlp(path: str) -> Dict[str, Any]:
    """
    Charge un modèle MLP depuis un fichier .gy.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du modèle (weights, biases, config).

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ModelFormatError: Si le format est invalide.
    """
    path = validate_model_path(path)
    header, tensors = load_model(path)
    model = merge_header(header, tensors)

    # Compatibilité : le format Trainer utilise les clés 'W' / 'b',
    # mais les fonctions predict attendent 'weights' / 'biases'.
    if 'weights' not in model and 'W' in model:
        model['weights'] = model.pop('W')
        model['biases'] = model.pop('b')

    logger.info("MLP chargé depuis %s", path)
    return model


def predict_mlp(model: Dict[str, Any], X) -> Dict[str, Any]:
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

    Raises:
        ValueError: Si X est vide ou malformé.
        KeyError: Si des clés requises manquent dans le modèle.
    """
    X = validate_ndarray(X, "X", min_dim_size=1)
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
                z_clipped = np.clip(z, -500, 500)
                a = 1.0 / (1.0 + np.exp(-z_clipped))  # sigmoid

    # Résultats
    if multiclass:
        predictions = a.argmax(axis=1)
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