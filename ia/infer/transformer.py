"""
IA/infer/transformer.py — Inférence des Transformers.
"""

import math
import pickle
import logging

import numpy as np

from ..cpp import get_core
C = get_core()

from ..train.transformer import MiniTransformer3D

logger = logging.getLogger(__name__)


# ==================================================================
# Helpers
# ==================================================================

def softmax(x):
    """Softmax stable le long du dernier axe."""
    return C.softmax(x)


def layer_norm(x, eps=1e-8):
    """Normalisation de couche le long du dernier axe."""
    return C.layer_norm(x, eps)


def relu(x):
    """Fonction d'activation ReLU."""
    return C.relu(x)


# ==================================================================
# MiniTransformer (single-head)
# ==================================================================

def load_transformer(path):
    """
    Charge un modèle Transformer single-head depuis un fichier pickle.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du modèle (embedding, W_q, W_k, W_v, etc.).
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Transformer chargé depuis %s", path)
    return model


def predict_transformer(model, tokens):
    """
    Prédiction binaire sur une séquence de tokens avec un Transformer single-head.

    Supporte deux formats :
      - Legacy (num_blocks absent ou 1) : clés W_q, W_k, W_v, W_ff1, b_ff1, …
      - Nouveau (num_blocks > 1) : clés W_q_list, W_k_list, W_v_list, …
        (listes de poids par bloc)

    Args:
        model: Dictionnaire de paramètres (issu de load_transformer).
        tokens: Liste d'entiers représentant les tokens.

    Returns:
        dict: {
            'probability': float,
            'class': str,
            'confidence': float,
            'attention_weights': ndarray,
        }
    """
    embedding = model['embedding']
    W_cls = model['W_cls']
    b_cls = model['b_cls']
    embed_dim = model['embed_dim']
    num_blocks = model.get('num_blocks', 1)

    token_indices = np.array(tokens)
    x_emb = embedding[token_indices]
    x = x_emb

    last_attn_weights = None
    for b_idx in range(num_blocks):
        if num_blocks == 1:
            wq = model['W_q']
            wk = model['W_k']
            wv = model['W_v']
            wff1 = model['W_ff1']
            bff1 = model['b_ff1']
            wff2 = model['W_ff2']
            bff2 = model['b_ff2']
            gamma_attn = model.get('gamma_attn')
            beta_attn = model.get('beta_attn')
            gamma_ff = model.get('gamma_ff')
            beta_ff = model.get('beta_ff')
        else:
            wq = model['W_q_list'][b_idx]
            wk = model['W_k_list'][b_idx]
            wv = model['W_v_list'][b_idx]
            wff1 = model['W_ff1_list'][b_idx]
            bff1 = model['b_ff1_list'][b_idx]
            wff2 = model['W_ff2_list'][b_idx]
            bff2 = model['b_ff2_list'][b_idx]
            gamma_attn = model.get('gamma_attn_list', [None] * num_blocks)[b_idx]
            beta_attn = model.get('beta_attn_list', [None] * num_blocks)[b_idx]
            gamma_ff = model.get('gamma_ff_list', [None] * num_blocks)[b_idx]
            beta_ff = model.get('beta_ff_list', [None] * num_blocks)[b_idx]

        # Self-attention
        Q = C.matmul(x, wq)
        K = C.matmul(x, wk)
        V = C.matmul(x, wv)
        attn_scores = C.matmul(Q, K.T) / math.sqrt(embed_dim)
        attention_weights = softmax(attn_scores)
        last_attn_weights = attention_weights
        attn_output = C.matmul(attention_weights, V)

        # Résiduel + LayerNorm (avec scale/shift appris si disponibles)
        x_attn = x + attn_output
        x_norm = layer_norm(x_attn)
        if gamma_attn is not None and beta_attn is not None:
            x = x_norm * gamma_attn + beta_attn
        else:
            x = x_norm

        # FFN
        ff1 = relu(C.matmul(x, wff1) + bff1)
        ff2 = C.matmul(ff1, wff2) + bff2

        # Résiduel + LayerNorm (avec scale/shift appris si disponibles)
        x_ff = x + ff2
        x_ff_norm = layer_norm(x_ff)
        if gamma_ff is not None and beta_ff is not None:
            x = x_ff_norm * gamma_ff + beta_ff
        else:
            x = x_ff_norm

    # Pooling moyen + classification
    pooled = np.array(C.mean_axis(x, 0)).reshape(1, -1)
    logits = C.matmul(pooled, W_cls) + b_cls
    probability = float(C.sigmoid(np.array([[logits[0, 0]]]))[0, 0])

    predicted_class = 'Classe 1' if probability > 0.5 else 'Classe 0'
    confidence = float(abs(probability - 0.5) * 2)

    logger.info("Transformer — probabilité: %.4f, classe: %s", probability, predicted_class)

    return {
        'probability': probability,
        'class': predicted_class,
        'confidence': confidence,
        'attention_weights': last_attn_weights,
    }


# ==================================================================
# MiniTransformer3D (multi-head)
# ==================================================================

def load_transformer3d(path):
    """
    Charge un modèle MiniTransformer3D multi-head depuis un fichier pickle.

    Utilise la méthode de classe MiniTransformer3D.load du module d'entraînement.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        MiniTransformer3D: Instance du modèle chargée.
    """
    model = MiniTransformer3D.load(path)
    logger.info("Transformer3D chargé depuis %s", path)
    return model


def predict_transformer3d(model, tokens):
    """
    Prédiction binaire sur une séquence de tokens avec un Transformer3D multi-head.

    Args:
        model: Instance de MiniTransformer3D (issu de load_transformer3d).
        tokens: Liste d'entiers représentant les tokens.

    Returns:
        dict: {
            'probability': float,
            'class': str,
            'confidence': float,
            'attention_weights': list,
        }
    """
    token_array = np.array(tokens).reshape(1, -1)
    out, cache = model.forward(token_array)

    probability = float(out[0, 0])
    predicted_class = 'Classe 1' if probability > 0.5 else 'Classe 0'
    confidence = float(abs(probability - 0.5) * 2)
    attention_weights = [aw.copy() for aw in cache['attn_weights']]

    logger.info("Transformer3D — probabilité: %.4f, classe: %s", probability, predicted_class)

    return {
        'probability': probability,
        'class': predicted_class,
        'confidence': confidence,
        'attention_weights': attention_weights,
    }