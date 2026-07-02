"""
IA/infer/rnn.py — Inférence du RNN.

Fonctions :
  - load_rnn(path)       : charge un modèle RNN depuis un fichier .gy
  - predict_rnn(model, seq): prédiction binaire sur une séquence

Usage :
    from IA import load_rnn, predict_rnn
    model = load_rnn("mini_rnn.gy")
    result = predict_rnn(model, sequence)
"""

import logging
from typing import Any, Dict, List

import numpy as np

from .._utils import merge_header, validate_model_path, validate_sequence
from ..cpp import get_core
from ..ia_format import load_model

C = get_core()

logger = logging.getLogger(__name__)


# ==================================================================
# Helpers
# ==================================================================

def sigmoid(x):
    """Fonction d'activation sigmoïde."""
    return C.sigmoid(x)


def tanh(x):
    """Fonction d'activation tanh."""
    return C.tanh(x)


# ==================================================================
# RNN
# ==================================================================

def load_rnn(path: str) -> Dict[str, Any]:
    """
    Charge un modèle RNN depuis un fichier .gy.

    Gère les formats V2 et V3.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du modèle.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    path = validate_model_path(path)
    header, tensors = load_model(path)
    model = merge_header(header, tensors)
    logger.info("RNN chargé depuis %s", path)
    return model


def predict_rnn(model: Dict[str, Any], sequence) -> Dict[str, Any]:
    """
    Prédiction binaire sur une séquence avec un RNN vanilla (mono ou multi-couche).

    Args:
        model: Dictionnaire de paramètres (issu de load_rnn).
        sequence: Tableau numpy de forme (seq_len, input_size).

    Returns:
        dict: {'probability', 'class', 'confidence', 'hidden_states'}
    """
    sequence = validate_sequence(sequence, "sequence")

    hidden_size = model['hidden_size']
    num_layers = model.get('num_layers', 1)
    W_hy = model['W_hy']
    b_y = model['b_y']

    seq_len = sequence.shape[0]
    h_list = [np.zeros((1, hidden_size)) for _ in range(num_layers)]
    hidden_states = [[h.copy() for h in h_list]]

    for t in range(seq_len):
        x_t = np.ascontiguousarray(sequence[t].reshape(1, -1))
        layer_input = x_t
        for l in range(num_layers):
            if num_layers == 1:
                W_xh = model['W_xh']
                W_hh = model['W_hh']
                b_h = model['b_h']
            else:
                W_xh = model[f'W_xh_{l}']
                W_hh = model[f'W_hh_{l}']
                b_h = model[f'b_h_{l}']
            h_list[l] = tanh(
                C.matmul(np.ascontiguousarray(layer_input), np.ascontiguousarray(W_xh))
                + C.matmul(np.ascontiguousarray(h_list[l]), np.ascontiguousarray(W_hh))
                + b_h
            )
            layer_input = np.ascontiguousarray(h_list[l])
        hidden_states.append([h.copy() for h in h_list])

    y_pred = sigmoid(C.matmul(h_list[-1], W_hy) + b_y)
    probability = float(y_pred[0, 0])
    predicted_class = 'Classe 1' if probability > 0.5 else 'Classe 0'
    confidence = float(abs(probability - 0.5) * 2)

    logger.info("RNN — probabilité: %.4f, classe: %s", probability, predicted_class)

    return {
        'probability': probability,
        'class': predicted_class,
        'confidence': confidence,
        'hidden_states': hidden_states,
    }