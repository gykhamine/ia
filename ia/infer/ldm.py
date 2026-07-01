"""
IA/infer/ldm.py — Inférence des modèles de diffusion (LDM).
"""

import math
import pickle
import logging

import numpy as np

from ..cpp import get_core
C = get_core()

logger = logging.getLogger(__name__)


# ==================================================================
# Helpers
# ==================================================================

def _is_new_ldm_format(model):
    """Détecte le nouveau format multi-couches (hidden_sizes != None)."""
    return model.get('hidden_sizes') is not None


def _predict_noise_from_params(model, x_noisy, class_id):
    """
    Reconstruit la passe avant du réseau de prédiction de bruit sauvegardé.

    Supporte deux formats :
      - Legacy (hidden_sizes=None) : clés W1, b1, W2, b2 (+ optionnel W3, b3)
        avec one-hot pour l'embedding de classe.
      - Nouveau (hidden_sizes=[...]) : clés W_0, b_0, …, W_{N-1}, b_{N-1}
        avec class_embedding appris.

    Args:
        model: Dictionnaire contenant les paramètres du réseau.
        x_noisy: Signal/image bruité, de forme quelconque.
        class_id: Identifiant de classe (int).

    Returns:
        ndarray: Bruit prédit, de même forme que x_noisy.
    """
    flat = x_noisy.flatten().reshape(1, -1)
    num_classes = model['num_classes']

    if _is_new_ldm_format(model):
        # Format nouveau : embedding appris + couches FC indexées
        class_embedding = model['class_embedding']
        c_embed = class_embedding[class_id % num_classes].reshape(1, -1)
        x_concat = np.concatenate([flat, c_embed], axis=1)

        # Déterminer le nombre de couches FC
        i = 0
        while f'W_{i}' in model:
            i += 1
        n_layers = i

        h = x_concat
        for layer_i in range(n_layers):
            W = model[f'W_{layer_i}']
            b = model[f'b_{layer_i}'].reshape(1, -1)
            z = C.matmul(h, W) + b
            if layer_i < n_layers - 1:
                h = C.relu(z)
            else:
                h = z  # pas de ReLU sur la dernière couche
        noise_flat = h
    else:
        # Format historique : one-hot + couches nommées
        class_onehot = np.zeros((1, num_classes))
        class_onehot[0, class_id % num_classes] = 1.0

        h = np.concatenate([flat, class_onehot], axis=1)
        h = C.relu(C.matmul(h, model['W1']) + model['b1'])
        h = C.relu(C.matmul(h, model['W2']) + model['b2'])
        if 'W3' in model:
            h = C.relu(C.matmul(h, model['W3']) + model['b3'])
        noise_flat = C.matmul(h, model['W_out']) + model['b_out']

    return noise_flat.reshape(x_noisy.shape)


# ==================================================================
# LDM Image
# ==================================================================

def load_ldm_image(path):
    """
    Charge un modèle de diffusion d'image depuis un fichier pickle.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du modèle (betas, alpha_cumprod, réseau de prédiction, etc.).
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("LDM image chargé depuis %s", path)
    return model


def generate_ldm_image(model, class_id, shape=(8, 8), num_steps=50):
    """
    Génère une image par débruitage DDPM itératif.

    Args:
        model: Dictionnaire de paramètres (issu de load_ldm_image).
        class_id: Identifiant de classe pour la génération conditionnelle.
        shape: Forme de l'image à générer (H, W).
        num_steps: Nombre d'étapes de débruitage.

    Returns:
        dict: {
            'generated': ndarray(shape),
            'class_id': int,
            'num_steps': int,
        }
    """
    betas = model['betas']
    alpha_cumprod = model['alpha_cumprod']

    x = np.random.randn(*shape)

    for t in reversed(range(num_steps)):
        predicted_noise = _predict_noise_from_params(model, x, class_id)

        alpha_t = alpha_cumprod[t]
        alpha_t_prev = alpha_cumprod[t - 1] if t > 0 else 1.0

        x_pred = (x - math.sqrt(1 - alpha_t) * predicted_noise) / math.sqrt(alpha_t)
        x = x_pred * math.sqrt(alpha_t_prev) + math.sqrt(1 - alpha_t_prev) * np.random.randn(*shape)

    logger.info("LDM image — classe: %d, forme: %s, étapes: %d", class_id, shape, num_steps)

    return {
        'generated': x,
        'class_id': int(class_id),
        'num_steps': int(num_steps),
    }


# ==================================================================
# LDM Audio
# ==================================================================

def load_ldm_audio(path):
    """
    Charge un modèle de diffusion audio depuis un fichier pickle.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du modèle.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("LDM audio chargé depuis %s", path)
    return model


def generate_ldm_audio(model, class_id, signal_length=64, num_steps=50):
    """
    Génère un signal audio par débruitage DDPM itératif.

    Args:
        model: Dictionnaire de paramètres (issu de load_ldm_audio).
        class_id: Identifiant de classe pour la génération conditionnelle.
        signal_length: Longueur du signal à générer.
        num_steps: Nombre d'étapes de débruitage.

    Returns:
        dict: {
            'generated': ndarray(signal_length),
            'class_id': int,
            'num_steps': int,
        }
    """
    betas = model['betas']
    alpha_cumprod = model['alpha_cumprod']
    shape = (signal_length,)

    x = np.random.randn(*shape)

    for t in reversed(range(num_steps)):
        predicted_noise = _predict_noise_from_params(model, x, class_id)

        alpha_t = alpha_cumprod[t]
        alpha_t_prev = alpha_cumprod[t - 1] if t > 0 else 1.0

        x_pred = (x - math.sqrt(1 - alpha_t) * predicted_noise) / math.sqrt(alpha_t)
        x = x_pred * math.sqrt(alpha_t_prev) + math.sqrt(1 - alpha_t_prev) * np.random.randn(*shape)

    logger.info("LDM audio — classe: %d, longueur: %d, étapes: %d", class_id, signal_length, num_steps)

    return {
        'generated': x,
        'class_id': int(class_id),
        'num_steps': int(num_steps),
    }