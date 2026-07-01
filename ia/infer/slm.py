"""
IA/infer/slm.py — Inférence des Small Language Models (SLM).
"""

import math
import pickle
import logging
import re

import numpy as np

from ..cpp import get_core
C = get_core()

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


def gelu(x):
    """Fonction d'activation GELU (approximation)."""
    return 0.5 * x * (1 + C.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


# ==================================================================
# TextPreprocessor (identique au module d'entraînement)
# ==================================================================

class TextPreprocessor:
    """Préprocesseur texte : tokenisation simple et décodage."""

    def __init__(self, vocab=None, max_len=32, default_vocab_size=500):
        self.vocab = vocab if vocab is not None else {}
        self.max_len = max_len
        self.default_vocab_size = default_vocab_size

    def tokenize(self, text):
        """
        Tokenise un texte en liste d'indices entiers.

        Tokens inconnus sont mappés à 1 (token <UNK>).

        Args:
            text: Chaîne de caractères à tokeniser.

        Returns:
            list[int]: Liste d'indices de tokens, tronquée à max_len.
        """
        words = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())
        token_ids = [self.vocab.get(w, 1) for w in words]
        return token_ids[:self.max_len]

    def decode_token(self, token_id):
        """
        Décode un identifiant de token en mot.

        Args:
            token_id: Identifiant entier du token.

        Returns:
            str: Mot correspondant, ou '<UNK>' si inconnu.
        """
        for word, idx in self.vocab.items():
            if idx == token_id:
                return word
        return '<UNK>'

    @classmethod
    def from_params(cls, params):
        """Reconstruit un TextPreprocessor depuis un dictionnaire de params."""
        p = cls()
        if isinstance(params, dict) and 'word_to_idx' in params:
            p.vocab = params['word_to_idx']
            p.vocab_size = params.get('vocab_size', len(p.vocab))
        return p


# ==================================================================
# Passe avant du SLM
# ==================================================================

def _slm_forward(params, token_indices):
    """
    Passe avant complète du SLM transformer.

    Supporte deux formats :
      - Legacy (num_blocks absent ou 1) : clés W_q, W_k, W_v, W_ff1, …
      - Nouveau (num_blocks > 1) : clés W_q_0, W_k_0, W_v_0, W_ff1_0, …
        (clés indexées par numéro de bloc, format de train/slm.py get_params)

    Reconstruit le graphe computationnel : embedding → [self-attention →
    LayerNorm → FFN → LayerNorm] * num_blocks → pooling → classification.

    Args:
        params: Dictionnaire contenant les poids du réseau.
        token_indices: Liste ou tableau d'indices de tokens.

    Returns:
        tuple: (probs, attn_weights)
            - probs: ndarray(num_classes,) — probabilités par classe.
            - attn_weights: ndarray(seq_len, seq_len) — poids d'attention.
    """
    embedding = params['embedding']
    num_blocks = params.get('num_blocks', 1)

    x_emb = embedding[token_indices]
    x = x_emb

    last_attn_weights = None
    for b_idx in range(num_blocks):
        if num_blocks == 1:
            W_q = params['W_q']
            W_k = params['W_k']
            W_v = params['W_v']
            gamma_attn = params.get('gamma_attn')
            beta_attn = params.get('beta_attn')
            W_ff1 = params['W_ff1']
            b_ff1 = params['b_ff1']
            W_ff2 = params['W_ff2']
            b_ff2 = params['b_ff2']
            gamma_ff = params.get('gamma_ff')
            beta_ff = params.get('beta_ff')
        else:
            W_q = params[f'W_q_{b_idx}']
            W_k = params[f'W_k_{b_idx}']
            W_v = params[f'W_v_{b_idx}']
            gamma_attn = params.get(f'gamma_attn_{b_idx}')
            beta_attn = params.get(f'beta_attn_{b_idx}')
            W_ff1 = params[f'W_ff1_{b_idx}']
            b_ff1 = params[f'b_ff1_{b_idx}']
            W_ff2 = params[f'W_ff2_{b_idx}']
            b_ff2 = params[f'b_ff2_{b_idx}']
            gamma_ff = params.get(f'gamma_ff_{b_idx}')
            beta_ff = params.get(f'beta_ff_{b_idx}')

        # Self-attention
        Q = C.matmul(x, W_q)
        K = C.matmul(x, W_k)
        V = C.matmul(x, W_v)
        attn_scores = C.matmul(Q, K.T) / math.sqrt(x.shape[-1])
        attn_weights = softmax(attn_scores)
        last_attn_weights = attn_weights
        attn_output = C.matmul(attn_weights, V)

        # Résiduel + LayerNorm (avec scale/shift appris)
        x_attn = x + attn_output
        x_norm = layer_norm(x_attn)
        if gamma_attn is not None:
            x = x_norm * gamma_attn + beta_attn
        else:
            x = x_norm

        # FFN
        ff1 = gelu(C.matmul(x, W_ff1) + b_ff1)
        ff2 = C.matmul(ff1, W_ff2) + b_ff2

        # Résiduel + LayerNorm (avec scale/shift appris)
        x_ff = x + ff2
        x_ff_norm = layer_norm(x_ff)
        if gamma_ff is not None:
            x = x_ff_norm * gamma_ff + beta_ff
        else:
            x = x_ff_norm

    # Pooling moyen + classification
    pooled = np.array(C.mean_axis(x, 0))
    if pooled.ndim == 1:
        pooled = pooled.reshape(1, -1)
    logits = C.matmul(pooled, params['W_cls']) + params['b_cls']
    probs = np.asarray(softmax(logits)).flatten()

    return probs, last_attn_weights


# ==================================================================
# SLM — Classification de texte
# ==================================================================

def load_slm(path):
    """
    Charge un modèle SLM depuis un fichier pickle.

    Le modèle contient les poids du réseau (network_params), les params
    du préprocesseur (preprocessor_params) et les noms de classes.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Dictionnaire avec clés 'network_params', 'preprocessor',
              'class_names' et métadonnées.
    """
    with open(path, 'rb') as f:
        raw = pickle.load(f)

    # Les poids sont sous 'network_params' (format train/slm.py)
    network_params = raw.get('network_params', raw.get('params', {}))
    class_names = raw.get('class_names', [])

    # Reconstruire le préprocesseur depuis preprocessor_params
    preprocessor_params = raw.get('preprocessor_params')
    if preprocessor_params is not None:
        preprocessor = TextPreprocessor.from_params(preprocessor_params)
    elif 'preprocessor' in raw:
        preprocessor = raw['preprocessor']
    else:
        preprocessor = TextPreprocessor()

    model = {
        'network_params': network_params,
        'preprocessor': preprocessor,
        'class_names': class_names,
    }
    for k in ('vocab_size', 'seq_len', 'embed_dim', 'ff_dim',
              'num_classes', 'num_blocks', 'accuracy'):
        if k in raw:
            model[k] = raw[k]

    logger.info("SLM chargé depuis %s", path)
    return model


def predict_slm(model, text):
    """
    Prédiction de classe sur un texte avec un SLM transformer.

    Args:
        model: Dictionnaire (issu de load_slm) contenant 'params',
               'preprocessor' et 'class_names'.
        text: Chaîne de caractères à classifier.

    Returns:
        dict: {
            'predictions': [{'class': str, 'probability': float}, ...],
            'attention_weights': ndarray,
        }
    """
    params = model.get('network_params', model.get('params', {}))
    preprocessor = model['preprocessor']
    class_names = model['class_names']

    token_indices = preprocessor.tokenize(text)

    if len(token_indices) == 0:
        token_indices = [1]

    token_array = np.array(token_indices)
    probs, attn_weights = _slm_forward(params, token_array)

    # Trier par probabilité décroissante
    sorted_indices = np.argsort(-probs)
    predictions = []
    for idx in sorted_indices:
        idx_int = int(idx)
        class_name = class_names[idx_int] if idx_int < len(class_names) else f'Classe {idx_int}'
        predictions.append({
            'class': class_name,
            'probability': float(probs[idx_int]),
        })

    top_class = predictions[0]['class']
    top_prob = predictions[0]['probability']
    logger.info("SLM — texte: '%s', prédiction: %s (%.4f)", text[:50], top_class, top_prob)

    return {
        'predictions': predictions,
        'attention_weights': attn_weights,
    }


# ==================================================================
# SLM — Prédiction du mot suivant
# ==================================================================

def predict_slm_next_word(model, context_tokens):
    """
    Prédit les mots les plus probables suite au contexte donné.

    Args:
        model: Dictionnaire (issu de load_slm) contenant 'params'
               et 'preprocessor'.
        context_tokens: Liste d'entiers représentant les tokens de contexte.

    Returns:
        dict: {
            'top_words': [{'word': str, 'probability': float}, ...],
        }
    """
    params = model.get('network_params', model.get('params', {}))
    preprocessor = model['preprocessor']

    if len(context_tokens) == 0:
        context_tokens = [0]

    token_array = np.array(context_tokens)
    probs, _ = _slm_forward(params, token_array)

    # Trier par probabilité décroissante, prendre le top-K
    top_k = min(10, len(probs))
    sorted_indices = np.argsort(-probs)[:top_k]

    top_words = []
    for idx in sorted_indices:
        idx_int = int(idx)
        word = preprocessor.decode_token(idx_int)
        top_words.append({
            'word': word,
            'probability': float(probs[idx_int]),
        })

    logger.info(
        "SLM next-word — contexte: %s, top: %s (%.4f)",
        context_tokens[:5],
        top_words[0]['word'] if top_words else 'N/A',
        top_words[0]['probability'] if top_words else 0.0,
    )

    return {
        'top_words': top_words,
    }