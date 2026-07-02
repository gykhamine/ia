"""
IA/train/cnn.py — Entraînement des CNN 2D et N-D.

Modèles :
  - CNN2D : classification d'images 5x5 binaire (croix, carré, diagonale, bordure, point)
  - CNNND : classification de volumes N-D (configurable)

Supporte un nombre configurable de couches de convolution via le paramètre
``num_conv_layers``.  Lorsque ``num_conv_layers=1`` (défaut), le comportement
est identique à la version originale (rétrocompatible).
"""

import math
import numpy as np
import os
import logging
from itertools import product as itertools_product

from ..config import get_config, MODELS_DIR, MODEL_EXTENSION, ensure_directories
from ..ia_format import save_model, serialize_model_dict
from ..cpp import get_core

C = get_core()

logger = logging.getLogger(__name__)


# ==================================================================
# Fonctions d'activation communes
# ==================================================================

def relu(x):
    return C.relu(x)

def relu_deriv(x):
    return C.relu_deriv(x)

def xavier_init(shape, seed=42):
    return C.xavier_init(tuple(shape), seed)


# ==================================================================
# Helpers pour la rétropropagation à travers plusieurs couches conv
# ==================================================================

def _conv_input_grad_2d(d_conv, kernel):
    """Calcule le gradient w.r.t. l'entrée d'une couche conv 2D.

    Équivalent à une convolution « pleine » de *d_conv* avec le noyau
    retourné à 180°.  Nécessaire pour propager le gradient à travers
    plusieurs couches de convolution empilées.
    """
    kshape = kernel.shape
    pad_width = tuple((s - 1, s - 1) for s in kshape)
    padded = np.pad(np.array(d_conv), pad_width, mode='constant')
    rotated = np.flip(np.array(kernel))
    return C.convolve2d(padded, rotated)


def _conv_input_grad_nd(d_conv, kernel):
    """Calcule le gradient w.r.t. l'entrée d'une couche conv N-D.

    Équivalent à une convolution « pleine » de *d_conv* avec le noyau
    retourné à 180°.  Nécessaire pour propager le gradient à travers
    plusieurs couches de convolution empilées.
    """
    kshape = kernel.shape
    pad_width = tuple((s - 1, s - 1) for s in kshape)
    padded = np.pad(np.array(d_conv), pad_width, mode='constant')
    rotated = np.flip(np.array(kernel))
    return C.convolve_nd(padded, rotated)


# ==================================================================
# CNN 2D
# ==================================================================

def create_2d_pattern(shape, pattern_type="cross"):
    """Crée des motifs 2D variés pour l'entraînement."""
    img = np.zeros(shape)
    center_y, center_x = shape[0] // 2, shape[1] // 2

    if pattern_type == "cross":
        img[center_y, :] = 1
        img[:, center_x] = 1
    elif pattern_type == "square":
        img[center_y-1:center_y+2, center_x-1:center_x+2] = 1
    elif pattern_type == "diagonal":
        for i in range(min(shape)):
            img[i, i] = 1
    elif pattern_type == "border":
        img[0, :] = 1
        img[-1, :] = 1
        img[:, 0] = 1
        img[:, -1] = 1
    elif pattern_type == "random":
        img = np.random.randint(0, 2, shape)
    elif pattern_type == "dot":
        img[center_y, center_x] = 1
    return img


def convolve2d(img, kernel):
    """Convolution 2D."""
    return C.convolve2d(img, kernel)


def convolve2d_backward(img, kernel, d_conv):
    """Rétropropagation pour la convolution 2D."""
    return C.convolve2d_backward(img, kernel, d_conv)


def train_cnn2d(X=None, y=None, input_shape=(5, 5), kernel_shape=(3, 3),
                lr=0.01, epochs=1000, early_stop_loss=0.001,
                save_path=None, seed=42, num_conv_layers=1):
    """
    Entraîne un CNN 2D pour la classification de motifs.

    Args:
        X: Données d'entrée (optionnel, générées automatiquement si None).
        y: Labels (optionnel, générés automatiquement si None).
        input_shape: Forme de l'image d'entrée (H, W).
        kernel_shape: Forme du kernel de convolution (utilisée pour toutes
                      les couches de convolution).
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques maximum.
        early_stop_loss: Seuil de loss pour arrêt précoce.
        save_path: Chemin de sauvegarde (défaut: models/mini_cnn.gy).
        seed: Graine aléatoire.
        num_conv_layers: Nombre de couches de convolution empilées
                         (défaut: 1, rétrocompatible).  Chaque couche
                         possède son propre kernel et biais.  La forme
                         du kernel est identique pour toutes les couches.

    Returns:
        dict: Paramètres du modèle entraîné et métriques.
    """
    ensure_directories()

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"mini_cnn{MODEL_EXTENSION}")

    # Données par défaut
    if X is None or y is None:
        X = np.array([
            create_2d_pattern(input_shape, "cross"),
            create_2d_pattern(input_shape, "random"),
            create_2d_pattern(input_shape, "square"),
            create_2d_pattern(input_shape, "diagonal"),
            create_2d_pattern(input_shape, "border"),
            create_2d_pattern(input_shape, "random"),
            create_2d_pattern(input_shape, "dot"),
            create_2d_pattern(input_shape, "cross"),
        ])
        y = np.array([[1], [0], [1], [1], [1], [0], [1], [1]])

    # Initialisation des couches de convolution
    kernels = []
    biases = []
    current_shape = input_shape
    for layer_i in range(num_conv_layers):
        k = xavier_init(kernel_shape, seed=seed + layer_i)
        b = C.zeros((1,))
        kernels.append(k)
        biases.append(b)
        current_shape = (current_shape[0] - kernel_shape[0] + 1,
                         current_shape[1] - kernel_shape[1] + 1)
    conv_shape = current_shape
    fc_input_size = math.prod(conv_shape)
    w_fc = xavier_init((fc_input_size, 1), seed=seed + num_conv_layers)
    b_fc = C.zeros((1,))

    best_loss = float('inf')
    best_model = None
    history = []

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            img = X[i]

            # Forward pass — couches de convolution empilées
            layer_inputs = []
            pre_activations = []
            x = img
            for layer_i in range(num_conv_layers):
                layer_inputs.append(x)
                conv = convolve2d(x, kernels[layer_i]) + biases[layer_i]
                pre_activations.append(conv)
                x = relu(conv)

            flat = x.flatten().reshape(1, -1)
            out = relu(C.matmul(flat, w_fc) + b_fc)
            error = y[i] - out
            loss = C.sum(C.pow(error, 2))
            total_loss += loss

            # Backward pass
            d_out = error * relu_deriv(out)
            d_flat = C.matmul(d_out, w_fc.T)
            d_w_fc = C.matmul(flat.T, d_out)
            d_b_fc = C.sum_axis(d_out, 0)
            d_x = d_flat.reshape(x.shape)

            # Rétropropagation à travers chaque couche conv en ordre inverse
            for layer_i in reversed(range(num_conv_layers)):
                d_conv = d_x * relu_deriv(pre_activations[layer_i])
                d_kernel = convolve2d_backward(layer_inputs[layer_i],
                                               kernels[layer_i], d_conv)
                d_bias = C.sum(d_conv)

                kernels[layer_i] = kernels[layer_i] + C.scale(d_kernel, lr)
                biases[layer_i] = biases[layer_i] + C.scale(d_bias, lr)

                # Propager le gradient vers la couche précédente
                if layer_i > 0:
                    d_x = _conv_input_grad_2d(d_conv, kernels[layer_i])

            w_fc = w_fc + C.scale(d_w_fc, lr)
            b_fc = b_fc + C.scale(d_b_fc, lr)

        avg_loss = total_loss / len(X)
        history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = {
                'kernels': [k.copy() for k in kernels],
                'biases': [b.copy() for b in biases],
                'w_fc': w_fc.copy(), 'b_fc': b_fc.copy(),
            }

        if epoch % 100 == 0:
            logger.info("Epoch %d, Loss: %.6f", epoch, avg_loss)

        if avg_loss < early_stop_loss:
            logger.info("Convergence atteinte à l'epoch %d", epoch)
            break

    # Restauration du meilleur modèle
    kernels = best_model['kernels']
    biases = best_model['biases']
    w_fc = best_model['w_fc']
    b_fc = best_model['b_fc']

    # Évaluation
    correct = 0
    for i in range(len(X)):
        img = X[i]
        x = img
        for layer_i in range(num_conv_layers):
            conv = convolve2d(x, kernels[layer_i]) + biases[layer_i]
            x = relu(conv)
        flat = x.flatten().reshape(1, -1)
        out = relu(C.matmul(flat, w_fc) + b_fc)
        prediction = 1 if out[0, 0] > 0.5 else 0
        y_val = y[i][0] if hasattr(y[i], '__len__') else y[i]
        if prediction == y_val:
            correct += 1
    accuracy = correct / len(X) * 100

    # Sauvegarde
    model = {
        'kernels': kernels, 'biases': biases,
        'w_fc': w_fc, 'b_fc': b_fc,
        'input_shape': input_shape, 'kernel_shape': kernel_shape,
        'conv_shape': conv_shape, 'accuracy': accuracy,
        'num_conv_layers': num_conv_layers,
    }
    header, tensors = serialize_model_dict(model)
    save_model(save_path, header, tensors)
    logger.info("CNN 2D sauvegardé dans %s (précision: %.1f%%)", save_path, accuracy)

    return {'model': model, 'save_path': save_path, 'accuracy': accuracy,
            'final_loss': best_loss, 'history': history}


# ==================================================================
# CNN N-D
# ==================================================================

def create_nd_volume(shape, pattern_type="center"):
    """Crée un volume ND avec différents motifs."""
    volume = np.zeros(shape)
    center = tuple(dim // 2 for dim in shape)

    if pattern_type == "center":
        if all(0 <= center[i] < shape[i] for i in range(len(shape))):
            volume[center] = 1
    elif pattern_type == "full":
        volume[:] = 1
    elif pattern_type == "random":
        volume = np.random.randint(0, 2, shape)
    elif pattern_type == "corner":
        corner = tuple(0 for _ in shape)
        volume[corner] = 1
    return volume


def convolve_nd(volume, kernel):
    """Convolution N-dimensionnelle générique."""
    return C.convolve_nd(volume, kernel)


def convolve_nd_backward(volume, kernel, d_conv):
    """Rétropropagation générique pour convolution ND."""
    return C.convolve_nd_backward(volume, kernel, d_conv)


def train_cnn_nd(dimensions=4, volume_shape=None, kernel_shape=None,
                 lr=0.01, epochs=1000, save_path=None, seed=42,
                 num_conv_layers=1):
    """
    Entraîne un CNN N-D pour la classification de volumes.

    Args:
        dimensions: Nombre de dimensions (2, 3, 4, 5...).
        volume_shape: Forme du volume (défaut: (3,3,3,3)[:dimensions]).
        kernel_shape: Forme du kernel (défaut: (2,2,2,2)[:dimensions]),
                      utilisée pour toutes les couches de convolution.
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques maximum.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        num_conv_layers: Nombre de couches de convolution empilées
                         (défaut: 1, rétrocompatible).  Chaque couche
                         possède son propre kernel et biais.  La forme
                         du kernel est identique pour toutes les couches.

    Returns:
        dict: Paramètres du modèle entraîné et métriques.
    """
    ensure_directories()

    if volume_shape is None:
        volume_shape = (3, 3, 3, 3)[:dimensions]
    if kernel_shape is None:
        kernel_shape = (2, 2, 2, 2)[:dimensions]
    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"mini_cnn_{dimensions}d{MODEL_EXTENSION}")

    # Données d'entraînement
    X = np.array([
        create_nd_volume(volume_shape, "center"),
        create_nd_volume(volume_shape, "full"),
        create_nd_volume(volume_shape, "random"),
        create_nd_volume(volume_shape, "corner"),
        create_nd_volume(volume_shape, "center"),
        create_nd_volume(volume_shape, "random"),
    ])
    y = np.array([[1], [1], [0], [1], [1], [0]])

    # Initialisation des couches de convolution
    kernels = []
    biases = []
    current_shape = volume_shape
    for layer_i in range(num_conv_layers):
        k = C.scale(C.randn(tuple(kernel_shape), seed + layer_i * 2), 0.1)
        b = C.scale(C.randn((1,), seed + layer_i * 2 + 1), 0.1)
        kernels.append(k)
        biases.append(b)
        current_shape = tuple(
            current_shape[d] - kernel_shape[d] + 1 for d in range(dimensions)
        )
    conv_shape = current_shape
    fc_input_size = math.prod(conv_shape)
    w_fc = C.scale(C.randn((fc_input_size, 1), seed + num_conv_layers * 2), 0.1)
    b_fc = C.scale(C.randn((1,), seed + num_conv_layers * 2 + 1), 0.1)

    history = []
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            volume = X[i]

            # Forward pass — couches de convolution empilées
            layer_inputs = []
            pre_activations = []
            x = volume
            for layer_i in range(num_conv_layers):
                layer_inputs.append(x)
                conv = convolve_nd(x, kernels[layer_i]) + biases[layer_i]
                pre_activations.append(conv)
                x = relu(conv)

            flat = x.flatten().reshape(1, -1)
            out = relu(C.matmul(flat, w_fc) + b_fc)
            error = y[i] - out
            loss = C.sum(C.pow(error, 2))
            total_loss += loss

            # Backward pass
            d_out = error * relu_deriv(out)
            d_flat = C.matmul(d_out, w_fc.T)
            d_w_fc = C.matmul(flat.T, d_out)
            d_b_fc = d_out
            d_x = d_flat.reshape(x.shape)

            # Rétropropagation à travers chaque couche conv en ordre inverse
            for layer_i in reversed(range(num_conv_layers)):
                d_conv = d_x * relu_deriv(pre_activations[layer_i])
                d_kernel = convolve_nd_backward(layer_inputs[layer_i],
                                                kernels[layer_i], d_conv)
                d_bias = C.sum(d_conv)

                kernels[layer_i] = kernels[layer_i] + C.scale(d_kernel, lr)
                biases[layer_i] = biases[layer_i] + C.scale(d_bias, lr)

                # Propager le gradient vers la couche précédente
                if layer_i > 0:
                    d_x = _conv_input_grad_nd(d_conv, kernels[layer_i])

            w_fc = w_fc + C.scale(d_w_fc, lr)
            b_fc = b_fc + C.scale(d_b_fc.ravel(), lr)

            kernels = [C.clip(k, -1, 1) for k in kernels]
            w_fc = C.clip(w_fc, -1, 1)

        avg_loss = total_loss / len(X)
        history.append(avg_loss)
        if epoch % 100 == 0:
            logger.info("Epoch %d, Loss: %.6f", epoch, avg_loss)

    # Évaluation
    correct = 0
    for i in range(len(X)):
        volume = X[i]
        x = volume
        for layer_i in range(num_conv_layers):
            conv = convolve_nd(x, kernels[layer_i]) + biases[layer_i]
            x = relu(conv)
        flat = x.flatten().reshape(1, -1)
        out = relu(C.matmul(flat, w_fc) + b_fc)
        prediction = 1 if out[0, 0] > 0.5 else 0
        y_val = y[i][0] if hasattr(y[i], '__len__') else y[i]
        if prediction == y_val:
            correct += 1
    accuracy = correct / len(X) * 100

    # Sauvegarde
    model = {
        'kernels': kernels, 'biases': biases,
        'w_fc': w_fc, 'b_fc': b_fc,
        'volume_shape': volume_shape, 'kernel_shape': kernel_shape,
        'dimensions': dimensions, 'conv_shape': conv_shape,
        'accuracy': accuracy, 'num_conv_layers': num_conv_layers,
    }
    header, tensors = serialize_model_dict(model)
    save_model(save_path, header, tensors)
    logger.info("CNN %dD sauvegardé dans %s (précision: %.1f%%)", dimensions, save_path, accuracy)

    return {'model': model, 'save_path': save_path, 'accuracy': accuracy, 'history': history}