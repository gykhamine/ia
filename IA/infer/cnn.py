"""
IA/infer/cnn.py — Inférence des CNN 2D et N-D.

Fonctions :
  - load_cnn2d / predict_cnn2d   : CNN 2D pour la classification de motifs.
  - load_cnn_nd / predict_cnn_nd : CNN N-D pour la classification de volumes.
"""

import logging
from itertools import product as itertools_product
from typing import Any, Dict, List, Optional

import numpy as np

from .._utils import merge_header, validate_model_path, validate_ndarray
from ..cpp import get_core
from ..ia_format import load_model

C = get_core()

logger = logging.getLogger(__name__)


# ==================================================================
# Helpers
# ==================================================================

def relu(x):
    """Fonction d'activation ReLU."""
    return C.relu(x)


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolution 2D (pure Python, équivalente au moteur C++)."""
    kh, kw = kernel.shape
    ih, iw = img.shape
    output = np.zeros((ih - kh + 1, iw - kw + 1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
    return output


def convolve_nd(volume: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolution N-dimensionnelle générique."""
    v_shape = np.array(volume.shape)
    k_shape = np.array(kernel.shape)
    out_shape = v_shape - k_shape + 1
    output = np.zeros(out_shape)
    for out_idx in itertools_product(*[range(dim) for dim in out_shape]):
        slices = tuple(slice(out_idx[i], out_idx[i] + k_shape[i]) for i in range(len(out_idx)))
        output[out_idx] = np.sum(volume[slices] * kernel)
    return output


def _resolve_kernel_list(model: Dict[str, Any], key: str) -> List[np.ndarray]:
    """Résout les kernels/biases depuis le modèle (pluriel, singulier, ou indexé)."""
    items = model.get(key)
    if items is not None and isinstance(items, list):
        return items
    singular = key.rstrip('s')  # 'kernels' -> 'kernel'
    val = model.get(singular)
    if val is not None:
        return [val]
    # Reconstruire depuis les clés indexées
    result = []
    i = 0
    while f'{key}_{i}' in model:
        result.append(model[f'{key}_{i}'])
        i += 1
    return result


# ==================================================================
# Load (partagé 2D / N-D)
# ==================================================================

def _load_cnn(path: str, nd_mode: bool = False) -> Dict[str, Any]:
    """Charge un modèle CNN (2D ou N-D) depuis un fichier .gy.

    Gère les formats V2 et V3.
    """
    path = validate_model_path(path)
    header, tensors = load_model(path)
    model = merge_header(header, tensors)

    # Convertir les shapes list→tuple pour volume_shape, image_shape, etc.
    for key in ('volume_shape', 'image_shape', 'input_shape', 'conv_shape'):
        if key in model and isinstance(model[key], list):
            model[key] = tuple(model[key])

    suffix = 'N-D' if nd_mode else '2D'
    logger.info("CNN %s chargé depuis %s", suffix, path)
    return model


# ==================================================================
# CNN 2D
# ==================================================================

def load_cnn2d(path: str) -> Dict[str, Any]:
    """Charge un modèle CNN 2D depuis un fichier .gy."""
    return _load_cnn(path, nd_mode=False)


def predict_cnn2d(model: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
    """
    Prédiction binaire sur une image 2D avec un CNN 2D.

    Args:
        model: Dictionnaire de paramètres (issu de load_cnn2d).
        image: Tableau numpy 2D (H, W) représentant l'image.

    Returns:
        dict: {'probability', 'class', 'confidence', 'activations'}
    """
    image = validate_ndarray(image, "image", expected_ndim=2)

    w_fc = model['w_fc']
    b_fc = model['b_fc']
    kernels = _resolve_kernel_list(model, 'kernels')
    biases = _resolve_kernel_list(model, 'biases')
    num_conv_layers = model['num_conv_layers']

    x = image
    for layer_i in range(num_conv_layers):
        if layer_i >= len(kernels) or layer_i >= len(biases):
            break
        conv = convolve2d(x, kernels[layer_i]) + biases[layer_i]
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

def load_cnn_nd(path: str) -> Dict[str, Any]:
    """Charge un modèle CNN N-D depuis un fichier .gy."""
    return _load_cnn(path, nd_mode=True)


def predict_cnn_nd(model: Dict[str, Any], volume: np.ndarray) -> Dict[str, Any]:
    """
    Prédiction binaire sur un volume N-D avec un CNN N-D.

    Args:
        model: Dictionnaire de paramètres (issu de load_cnn_nd).
        volume: Tableau numpy N-D représentant le volume d'entrée.

    Returns:
        dict: {'probability', 'class', 'confidence', 'activations'}
    """
    volume = validate_ndarray(volume, "volume", min_dim_size=1)

    w_fc = model['w_fc']
    b_fc = model['b_fc']
    kernels = _resolve_kernel_list(model, 'kernels')
    biases = _resolve_kernel_list(model, 'biases')
    num_conv_layers = model['num_conv_layers']

    x = volume
    for layer_i in range(num_conv_layers):
        if layer_i >= len(kernels) or layer_i >= len(biases):
            break
        conv = convolve_nd(x, kernels[layer_i]) + biases[layer_i]
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