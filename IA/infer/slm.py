"""
IA/infer/slm.py — Inférence des Small Language Models (SLM).

Fonctions :
  - load_slm / predict_slm           : Classification de texte
  - predict_slm_next_word            : Prédiction du mot suivant
"""

import math
import logging
import re
from typing import Any, Dict, List

import numpy as np

from .._utils import validate_model_path
from ..cpp import get_core
from ..ia_format import load_model

C = get_core()

logger = logging.getLogger(__name__)


# ==================================================================
# Helpers
# ==================================================================

def softmax(x):
    return C.softmax(x)

def layer_norm(x, eps=1e-8):
    return C.layer_norm(x, eps)

def gelu(x):
    return 0.5 * x * (1 + C.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


class TextPreprocessor:
    """Préprocesseur texte : tokenisation simple et décodage."""

    def __init__(self, vocab=None, max_len=32, default_vocab_size=500):
        self.vocab = vocab if vocab is not None else {}
        self.max_len = max_len
        self.default_vocab_size = default_vocab_size

    def tokenize(self, text: str) -> List[int]:
        """Tokenise un texte en liste d'indices entiers."""
        if not text or not text.strip():
            return [1]  # <UNK>
        words = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())
        token_ids = [self.vocab.get(w, 1) for w in words]
        return token_ids[:self.max_len]

    def decode_token(self, token_id: int) -> str:
        """Décode un identifiant de token en mot."""
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

def _slm_forward(params: Dict[str, Any], token_indices: np.ndarray):
    """Passe avant complète du SLM transformer."""
    embedding = np.array(params['embedding'])
    num_blocks = params.get('num_blocks', 1)

    x_emb = embedding[np.array(token_indices, dtype=np.int64)]
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

        Q = C.matmul(x, W_q)
        K = C.matmul(x, W_k)
        V = C.matmul(x, W_v)
        attn_scores = C.matmul(Q, K.T) / math.sqrt(x.shape[-1])
        attn_weights = softmax(attn_scores)
        last_attn_weights = attn_weights
        attn_output = C.matmul(attn_weights, V)

        x_attn = x + attn_output
        x_norm = layer_norm(x_attn)
        if gamma_attn is not None:
            x = x_norm * gamma_attn + (beta_attn if beta_attn is not None else 0.0)
        else:
            x = x_norm

        ff1 = gelu(C.matmul(x, W_ff1) + b_ff1)
        ff2 = C.matmul(ff1, W_ff2) + b_ff2

        x_ff = x + ff2
        x_ff_norm = layer_norm(x_ff)
        if gamma_ff is not None:
            x = x_ff_norm * gamma_ff + (beta_ff if beta_ff is not None else 0.0)
        else:
            x = x_ff_norm

    pooled = np.array(C.mean_axis(x, 0))
    if pooled.ndim == 1:
        pooled = pooled.reshape(1, -1)
    logits = C.matmul(pooled, params['W_cls']) + params['b_cls']
    probs = np.asarray(softmax(logits)).flatten()

    return probs, last_attn_weights


# ==================================================================
# SLM — Classification de texte
# ==================================================================

def load_slm(path: str) -> Dict[str, Any]:
    """Charge un modèle SLM depuis un fichier .gy (format V2/V3)."""
    path = validate_model_path(path)
    header, tensors = load_model(path)

    v3_config = header.get('config', {})
    config = dict(header)
    config.update(v3_config)
    config.pop('config', None)

    network_params = {}
    for k, v in tensors.items():
        if not k.startswith('_'):
            network_params[k] = v

    if not network_params:
        raw_params = config.get('network_params', {})
        for k, v in raw_params.items():
            if isinstance(v, list):
                network_params[k] = np.array(v, dtype=np.float64)
            else:
                network_params[k] = v

    # Reconstruct list-type weights
    list_bases = set()
    for k in list(network_params.keys()):
        if '_' in k:
            base, idx = k.rsplit('_', 1)
            if idx.isdigit():
                list_bases.add(base)
    for base in sorted(list_bases):
        indexed = sorted(
            [(int(k.rsplit('_', 1)[1]), network_params[k])
             for k in network_params if k.startswith(base + '_')
             and k[len(base)+1:].isdigit()],
            key=lambda x: x[0]
        )
        if indexed:
            network_params[base] = [v for _, v in indexed]

    class_names = config.get('class_names', [])
    preprocessor_params = config.get('preprocessor_params')
    model = {'network_params': network_params, 'class_names': class_names}
    for k in ('vocab_size', 'seq_len', 'embed_dim', 'ff_dim',
              'num_classes', 'num_blocks', 'accuracy'):
        if k in config:
            model[k] = config[k]
    if preprocessor_params is not None:
        model['preprocessor'] = TextPreprocessor.from_params(preprocessor_params)
    else:
        model['preprocessor'] = TextPreprocessor()
    logger.info("SLM chargé depuis %s", path)
    return model


def predict_slm(model: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Prédiction de classe sur un texte avec un SLM transformer."""
    if not text or not text.strip():
        raise ValueError("Le texte ne peut pas être vide")

    params = model.get('network_params', model.get('params', {}))
    preprocessor = model['preprocessor']
    class_names = model['class_names']

    token_indices = preprocessor.tokenize(text)
    if len(token_indices) == 0:
        token_indices = [1]

    token_array = np.array(token_indices)
    probs, attn_weights = _slm_forward(params, token_array)

    sorted_indices = np.argsort(-probs)
    predictions = []
    for idx in sorted_indices:
        idx_int = int(idx)
        class_name = class_names[idx_int] if idx_int < len(class_names) else f'Classe {idx_int}'
        predictions.append({'class': class_name, 'probability': float(probs[idx_int])})

    top_class = predictions[0]['class']
    top_prob = predictions[0]['probability']
    logger.info("SLM — texte: '%s', prédiction: %s (%.4f)", text[:50], top_class, top_prob)

    return {'predictions': predictions, 'attention_weights': attn_weights}


# ==================================================================
# SLM — Prédiction du mot suivant
# ==================================================================

def predict_slm_next_word(model: Dict[str, Any],
                           context_tokens: List[int]) -> Dict[str, Any]:
    """Prédit les mots les plus probables suite au contexte donné."""
    if not context_tokens:
        context_tokens = [0]

    params = model.get('network_params', model.get('params', {}))
    preprocessor = model['preprocessor']

    token_array = np.array(context_tokens)
    probs, _ = _slm_forward(params, token_array)

    top_k = min(10, len(probs))
    sorted_indices = np.argsort(-probs)[:top_k]

    top_words = []
    for idx in sorted_indices:
        idx_int = int(idx)
        word = preprocessor.decode_token(idx_int)
        top_words.append({'word': word, 'probability': float(probs[idx_int])})

    logger.info("SLM next-word — contexte: %s, top: %s (%.4f)",
                context_tokens[:5],
                top_words[0]['word'] if top_words else 'N/A',
                top_words[0]['probability'] if top_words else 0.0)

    return {'top_words': top_words}