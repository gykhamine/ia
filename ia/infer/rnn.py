"""
IA/infer/rnn.py — Inférence du RNN.
"""

import pickle
import logging

import numpy as np

from ..cpp import get_core
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

def load_rnn(path):
    """
    Charge un modèle RNN depuis un fichier pickle.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du modèle (W_xh, W_hh, b_h, W_hy, b_y, etc.).
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("RNN chargé depuis %s", path)
    return model


def predict_rnn(model, sequence):
    """
    Prédiction binaire sur une séquence avec un RNN vanilla (mono ou multi-couche).

    Supporte deux formats :
      - Legacy (num_layers absent ou 1) : clés W_xh, W_hh, b_h
      - Nouveau (num_layers > 1) : clés W_xh_0, W_hh_0, b_h_0, …, W_xh_{N-1}, …

    Args:
        model: Dictionnaire de paramètres (issu de load_rnn).
        sequence: Tableau numpy de forme (seq_len, input_size).

    Returns:
        dict: {
            'probability': float,
            'class': str ('Classe 1' ou 'Classe 0'),
            'confidence': float,
            'hidden_states': list,
        }
    """
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