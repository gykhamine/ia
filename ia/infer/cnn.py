"""
IA/infer/cnn.py — Inférence des CNN 2D et N-D.
"""

import pickle
import logging
from itertools import product as itertools_product

import numpy as np

from ..cpp import get_core
C = get_core()

logger = logging.getLogger(__name__)


# ==================================================================
# Helpers
# ==================================================================

def relu(x):
    """Fonction d'activation ReLU."""
    return C.relu(x)


def convolve2d(img, kernel):
    """Convolution 2D (sans biais)."""
    kh, kw = kernel.shape
    ih, iw = img.shape
    output = np.zeros((ih - kh + 1, iw - kw + 1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
    return output


def convolve_nd(volume, kernel):
    """Convolution N-dimensionnelle générique."""
    v_shape = np.array(volume.shape)
    k_shape = np.array(kernel.shape)
    out_shape = v_shape - k_shape + 1
    output = np.zeros(out_shape)
    for out_idx in itertools_product(*[range(dim) for dim in out_shape]):
        slices = tuple(slice(out_idx[i], out_idx[i] + k_shape[i]) for i in range(len(out_idx)))
        output[out_idx] = np.sum(volume[slices] * kernel)
    return output


# ==================================================================
# CNN 2D
# ==================================================================

def load_cnn2d(path):
    """
    Charge un modèle CNN 2D depuis un fichier pickle.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du modèle (kernel, bias, w_fc, b_fc, etc.).
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("CNN 2D chargé depuis %s", path)
    return model


def _is_new_cnn_format(model):
    """Détecte le nouveau format multi-couches (clé 'kernels' présente)."""
    return 'kernels' in model


def predict_cnn2d(model, image):
    """
    Prédiction binaire sur une image 2D avec un CNN 2D.

    Supporte deux formats :
      - Legacy (num_conv_layers=1) : clés kernel, bias
      - Nouveau (num_conv_layers>1) : clés kernels, biases, num_conv_layers

    Args:
        model: Dictionnaire de paramètres (issu de load_cnn2d).
        image: Tableau numpy 2D (H, W) représentant l'image.

    Returns:
        dict: {
            'probability': float,
            'class': str ('STRUCTURÉ' ou 'ALÉATOIRE'),
            'confidence': float,
            'activations': ndarray,
        }
    """
    w_fc = model['w_fc']
    b_fc = model['b_fc']

    if _is_new_cnn_format(model):
        kernels = model['kernels']
        biases = model['biases']
        num_conv_layers = model['num_conv_layers']
        x = image
        for layer_i in range(num_conv_layers):
            conv = convolve2d(x, kernels[layer_i]) + biases[layer_i]
            x = relu(conv)
    else:
        kernel = model['kernel']
        bias = model['bias']
        conv = convolve2d(image, kernel) + bias
        x = relu(conv)

    activations = x
    flat = x.flatten().reshape(1, -1)
    out = C.relu(C.matmul(flat, w_fc) + b_fc)

    probability = float(out[0, 0])
    predicted_class = 'STRUCTURÉ' if probability > 0.5 else 'ALÉATOIRE'
    confidence = float(abs(probability - 0.5) * 2)

    logger.info("CNN 2D — probabilité: %.4f, classe: %s", probability, predicted_class)

    return {
        'probability': probability,
        'class': predicted_class,
        'confidence': confidence,
        'activations': activations,
    }


# ==================================================================
# CNN N-D
# ==================================================================

def load_cnn_nd(path):
    """
    Charge un modèle CNN N-D depuis un fichier pickle.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du modèle.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("CNN N-D chargé depuis %s", path)
    return model


def predict_cnn_nd(model, volume):
    """
    Prédiction binaire sur un volume N-D avec un CNN N-D.

    Supporte deux formats :
      - Legacy (num_conv_layers=1) : clés kernel, bias
      - Nouveau (num_conv_layers>1) : clés kernels, biases, num_conv_layers

    Args:
        model: Dictionnaire de paramètres (issu de load_cnn_nd).
        volume: Tableau numpy N-D représentant le volume d'entrée.

    Returns:
        dict: {
            'probability': float,
            'class': str,
            'confidence': float,
            'activations': ndarray,
        }
    """
    w_fc = model['w_fc']
    b_fc = model['b_fc']

    if _is_new_cnn_format(model):
        kernels = model['kernels']
        biases = model['biases']
        num_conv_layers = model['num_conv_layers']
        x = volume
        for layer_i in range(num_conv_layers):
            conv = convolve_nd(x, kernels[layer_i]) + biases[layer_i]
            x = relu(conv)
    else:
        kernel = model['kernel']
        bias = model['bias']
        conv = convolve_nd(volume, kernel) + bias
        x = relu(conv)

    activations = x
    flat = x.flatten().reshape(1, -1)
    out = C.relu(C.matmul(flat, w_fc) + b_fc)

    probability = float(out[0, 0])
    predicted_class = 'STRUCTURÉ' if probability > 0.5 else 'ALÉATOIRE'
    confidence = float(abs(probability - 0.5) * 2)

    logger.info("CNN N-D — probabilité: %.4f, classe: %s", probability, predicted_class)

    return {
        'probability': probability,
        'class': predicted_class,
        'confidence': confidence,
        'activations': activations,
    }