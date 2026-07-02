"""
IA/train/slm.py — Entraînement des Small Language Models (SLM).

Modèles :
  - train_slm_next_word : Prédiction du mot suivant (vocabulaire français, 10 mots).
  - train_slm_emotion : Détection d'émotion (6 classes : joie, tristesse, colère, peur, surprise, neutre).
  - train_slm_mood : Détection d'humeur (8 classes : joyeux, triste, énergique, calmé, stressé, fatigué, motivé, anxieux).
  - train_slm_statement : Classification de type de phrase (5 classes : question, affirmation, ordre, conseil, exclamation).
  - train_slm_sentiment : Analyse de sentiment (3 classes : positif, négatif, neutre).
"""

import math
import numpy as np
import os
import logging

from ..config import MODELS_DIR, MODEL_EXTENSION, ensure_directories
from ..cpp import get_core
from ..ia_format import save_model, serialize_model_dict

C = get_core()

logger = logging.getLogger(__name__)

class _SeedGenerator:
    """Générateur de seeds pour les appels C++ (SLM)."""
    __slots__ = ('_counter',)
    def __init__(self, start=0):
        self._counter = start
    def next(self):
        s = self._counter
        self._counter += 1
        return s

_seed_gen = _SeedGenerator()

def _next_seed():
    return _seed_gen.next()


# ==================================================================
# Fonctions communes
# ==================================================================

def softmax(x):
    return C.softmax(x)


def layer_norm(x, eps=1e-8):
    return C.layer_norm(x, eps)


def gelu(x):
    return C.gelu(x)


def gelu_deriv(x):
    return C.gelu_deriv(x)


def xavier_init(shape):
    return C.xavier_init(tuple(shape), _next_seed())


# ==================================================================
# TextPreprocessor
# ==================================================================

class TextPreprocessor:
    """Préprocesseur de texte simple : tokenisation whitespace + padding."""

    def __init__(self):
        self.word_to_idx = {}
        self.vocab_size = 0

    def fit(self, texts):
        """Construit le vocabulaire à partir d'une liste de textes."""
        all_words = set()
        for text in texts:
            all_words.update(self._tokenize(text))
        self.word_to_idx = {w: i + 2 for i, w in enumerate(sorted(all_words))}
        self.word_to_idx['PAD'] = 0
        self.word_to_idx['UNK'] = 1
        self.vocab_size = len(self.word_to_idx)

    def _tokenize(self, text):
        """Tokenisation simple : espace + minuscules."""
        return text.lower().split()

    def encode(self, text, seq_len):
        """Tokenise et pad/tronque à seq_len."""
        tokens = self._tokenize(text)
        indices = [self.word_to_idx.get(t, 1) for t in tokens]
        if len(indices) < seq_len:
            indices += [0] * (seq_len - len(indices))
        else:
            indices = indices[:seq_len]
        return np.array(indices)

    def get_params(self):
        return {'word_to_idx': self.word_to_idx, 'vocab_size': self.vocab_size}

    @classmethod
    def from_params(cls, params):
        p = cls()
        p.word_to_idx = params['word_to_idx']
        p.vocab_size = params['vocab_size']
        return p


# ==================================================================
# SLMClassifier
# ==================================================================

class SLMClassifier:
    """Classifieur SLM : Embedding -> [Self-Attention -> Add&Norm -> FFN(GELU) -> Add&Norm] * num_blocks -> Pool -> Softmax.

    Supports configurable depth via ``num_blocks`` (default 1).

    - When ``num_blocks=1``: parameters are stored with flat names
      (``W_q``, ``W_k``, …) for backward-compatible serialisation.
    - When ``num_blocks>1``: each block's parameters are stored in
      indexed lists (``_W_q[i]``, ``_K_k[i]``, …).
    """

    # Parameter key groups used per transformer block
    _BLOCK_PARAM_NAMES = [
        'W_q', 'W_k', 'W_v',
        'gamma_attn', 'beta_attn',
        'gamma_ff', 'beta_ff',
        'W_ff1', 'b_ff1', 'W_ff2', 'b_ff2',
    ]

    def __init__(self, vocab_size, seq_len, embed_dim, num_classes,
                 ff_dim=None, num_blocks=1, seed=42):
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.ff_dim = ff_dim if ff_dim is not None else embed_dim * 2

        self.embedding = xavier_init((vocab_size, embed_dim))

        # ----- Block parameters (always stored as lists internally) -----
        self._W_q = [xavier_init((embed_dim, embed_dim)) for _ in range(num_blocks)]
        self._W_k = [xavier_init((embed_dim, embed_dim)) for _ in range(num_blocks)]
        self._W_v = [xavier_init((embed_dim, embed_dim)) for _ in range(num_blocks)]
        self._gamma_attn = [C.ones((1, embed_dim)) for _ in range(num_blocks)]
        self._beta_attn = [C.zeros((1, embed_dim)) for _ in range(num_blocks)]
        self._gamma_ff = [C.ones((1, embed_dim)) for _ in range(num_blocks)]
        self._beta_ff = [C.zeros((1, embed_dim)) for _ in range(num_blocks)]
        self._W_ff1 = [xavier_init((embed_dim, self.ff_dim)) for _ in range(num_blocks)]
        self._b_ff1 = [C.zeros((1, self.ff_dim)) for _ in range(num_blocks)]
        self._W_ff2 = [xavier_init((self.ff_dim, embed_dim)) for _ in range(num_blocks)]
        self._b_ff2 = [C.zeros((1, embed_dim)) for _ in range(num_blocks)]

        # Backward-compatible flat attributes when num_blocks == 1
        if num_blocks == 1:
            self.W_q = self._W_q[0]
            self.W_k = self._W_k[0]
            self.W_v = self._W_v[0]
            self.gamma_attn = self._gamma_attn[0]
            self.beta_attn = self._beta_attn[0]
            self.gamma_ff = self._gamma_ff[0]
            self.beta_ff = self._beta_ff[0]
            self.W_ff1 = self._W_ff1[0]
            self.b_ff1 = self._b_ff1[0]
            self.W_ff2 = self._W_ff2[0]
            self.b_ff2 = self._b_ff2[0]

        # Classification head (shared across all blocks)
        self.W_cls = xavier_init((embed_dim, num_classes))
        self.b_cls = C.zeros((1, num_classes))
        self._cache = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x, pad_mask=None):
        """x: (seq_len,) token indices. Returns (num_classes,) softmax probabilities."""
        x_emb = self.embedding[x]  # (seq_len, embed_dim)

        block_caches = []
        h = x_emb
        for i in range(self.num_blocks):
            W_q = self._W_q[i]
            W_k = self._W_k[i]
            W_v = self._W_v[i]
            gamma_attn = self._gamma_attn[i]
            beta_attn = self._beta_attn[i]
            gamma_ff = self._gamma_ff[i]
            beta_ff = self._beta_ff[i]
            W_ff1 = self._W_ff1[i]
            b_ff1 = self._b_ff1[i]
            W_ff2 = self._W_ff2[i]
            b_ff2 = self._b_ff2[i]

            # Self-attention
            Q = C.matmul(h, W_q)
            K = C.matmul(h, W_k)
            V = C.matmul(h, W_v)
            attn_scores = C.matmul(Q, K.T) / math.sqrt(self.embed_dim)

            if pad_mask is not None:
                mask_2d = pad_mask[:, None] * pad_mask[None, :]
                attn_scores = attn_scores + (~mask_2d).astype(float) * (-1e9)

            attn_weights = C.softmax(attn_scores)
            attn_output = C.matmul(attn_weights, V)

            # Add & Norm (attention)
            x_attn = h + attn_output
            x_ln1 = C.layer_norm(x_attn)
            h_attn_norm = x_ln1 * gamma_attn + beta_attn

            # FFN (GELU)
            ff_hidden = C.matmul(h_attn_norm, W_ff1) + b_ff1
            ff_act = C.gelu(ff_hidden)
            ff_out = C.matmul(ff_act, W_ff2) + b_ff2

            # Add & Norm (FFN)
            x_ff = h_attn_norm + ff_out
            x_ln2 = C.layer_norm(x_ff)
            h_ff_norm = x_ln2 * gamma_ff + beta_ff

            block_caches.append({
                'h_in': h, 'Q': Q, 'K': K, 'V': V,
                'attn_weights': attn_weights, 'attn_output': attn_output,
                'h_attn_norm': h_attn_norm,
                'ff_hidden': ff_hidden, 'ff_act': ff_act, 'ff_out': ff_out,
                'h_ff_norm': h_ff_norm,
            })
            h = h_ff_norm

        # Mean pooling (on output of last block)
        if pad_mask is not None:
            mask_exp = pad_mask[:, None].astype(float)
            x_masked = h * mask_exp
            valid_count = max(np.sum(pad_mask), 1)
            pooled = C.sum_axis(x_masked, 0).reshape(1, -1) / valid_count
        else:
            pooled = C.mean_axis(h, 0).reshape(1, -1)

        # Classification
        logits = C.matmul(pooled, self.W_cls) + self.b_cls
        out = C.softmax(logits)[0]

        self._cache = {
            'x': x, 'pad_mask': pad_mask,
            'block_caches': block_caches,
            'pooled': pooled,
        }
        return out

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def backward(self, x, y, out, lr=0.01):
        """Backprop through full network. y: integer class label."""
        c = self._cache

        # Cross-entropy gradient
        d_logits = out.copy()
        d_logits[y] -= 1.0

        # Classification head
        d_W_cls = C.matmul(c['pooled'].T, d_logits.reshape(1, -1))
        d_b_cls = d_logits.reshape(1, -1)
        d_pooled = C.matmul(d_logits.reshape(1, -1), self.W_cls.T)

        # Mean pooling backward
        if c['pad_mask'] is not None:
            valid_count = max(np.sum(c['pad_mask']), 1)
            d_h = C.tile(d_pooled / valid_count, self.seq_len)
            d_h = d_h * c['pad_mask'][:, None].astype(float)
        else:
            d_h = C.tile(d_pooled / self.seq_len, self.seq_len)

        # Loop over blocks in reverse
        for i in reversed(range(self.num_blocks)):
            bc = c['block_caches'][i]

            W_ff1 = self._W_ff1[i]
            W_ff2 = self._W_ff2[i]
            b_ff1 = self._b_ff1[i]
            b_ff2 = self._b_ff2[i]
            W_q = self._W_q[i]
            W_k = self._W_k[i]
            W_v = self._W_v[i]

            # Add & Norm (FFN) backward — simplified
            d_h_attn_norm = d_h.copy()
            d_ff_out = d_h.copy()

            # FFN backward
            d_ff_act = C.matmul(d_ff_out, W_ff2.T)
            d_W_ff2_i = C.matmul(bc['ff_act'].T, d_ff_out)
            d_b_ff2_i = C.sum_axis(d_ff_out, 0).reshape(1, -1)
            d_ff_hidden = d_ff_act * C.gelu_deriv(bc['ff_hidden'])
            d_W_ff1_i = C.matmul(bc['h_attn_norm'].T, d_ff_hidden)
            d_b_ff1_i = C.sum_axis(d_ff_hidden, 0).reshape(1, -1)
            d_h_attn_norm = d_h_attn_norm + C.matmul(d_ff_hidden, W_ff1.T)

            # Add & Norm (attention) backward — simplified
            d_h_in = d_h_attn_norm.copy()
            d_attn_output = d_h_attn_norm.copy()

            # Attention backward
            d_V = C.matmul(bc['attn_weights'].T, d_attn_output)
            d_attn_weights = C.matmul(d_attn_output, bc['V'].T)
            inner = bc['attn_weights'] * d_attn_weights
            sum_inner = C.sum_axis(inner, -1).reshape(-1, 1)
            d_attn_scores = bc['attn_weights'] * (d_attn_weights - sum_inner)
            d_attn_scores = d_attn_scores / math.sqrt(self.embed_dim)
            d_Q = C.matmul(d_attn_scores, bc['K'])
            d_K = C.matmul(d_attn_scores.T, bc['Q'])

            dW_q_i = C.matmul(bc['h_in'].T, d_Q)
            dW_k_i = C.matmul(bc['h_in'].T, d_K)
            dW_v_i = C.matmul(bc['h_in'].T, d_V)
            d_h_in_q = C.matmul(d_Q, W_q.T)
            d_h_in_k = C.matmul(d_K, W_k.T)
            d_h_in_v = C.matmul(d_V, W_v.T)
            d_h = d_h_in_q + d_h_in_k + d_h_in_v + d_h_in

            # Update block i parameters (gradient DESCENT: W -= lr * grad)
            self._W_q[i] = self._W_q[i] - lr * dW_q_i
            self._W_k[i] = self._W_k[i] - lr * dW_k_i
            self._W_v[i] = self._W_v[i] - lr * dW_v_i
            self._W_ff1[i] = self._W_ff1[i] - lr * d_W_ff1_i
            self._b_ff1[i] = self._b_ff1[i] - lr * d_b_ff1_i
            self._W_ff2[i] = self._W_ff2[i] - lr * d_W_ff2_i
            self._b_ff2[i] = self._b_ff2[i] - lr * d_b_ff2_i

        # Update embedding
        for j, token_id in enumerate(c['x']):
            self.embedding[token_id] = self.embedding[token_id] - lr * d_h[j]

        # Update classification head
        self.W_cls = self.W_cls - lr * d_W_cls
        self.b_cls = self.b_cls - lr * d_b_cls

        # Clip block weights
        for i in range(self.num_blocks):
            self._W_q[i] = C.clip(self._W_q[i], -1, 1)
            self._W_k[i] = C.clip(self._W_k[i], -1, 1)
            self._W_v[i] = C.clip(self._W_v[i], -1, 1)
            self._W_ff1[i] = C.clip(self._W_ff1[i], -1, 1)
            self._W_ff2[i] = C.clip(self._W_ff2[i], -1, 1)
        self.W_cls = C.clip(self.W_cls, -1, 1)

        # Sync flat attributes for backward compatibility when num_blocks == 1
        if self.num_blocks == 1:
            self.W_q = self._W_q[0]
            self.W_k = self._W_k[0]
            self.W_v = self._W_v[0]
            self.gamma_attn = self._gamma_attn[0]
            self.beta_attn = self._beta_attn[0]
            self.gamma_ff = self._gamma_ff[0]
            self.beta_ff = self._beta_ff[0]
            self.W_ff1 = self._W_ff1[0]
            self.b_ff1 = self._b_ff1[0]
            self.W_ff2 = self._W_ff2[0]
            self.b_ff2 = self._b_ff2[0]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def get_params(self):
        """Return a dict of all model parameters.

        For ``num_blocks=1`` the block parameters use the original flat
        key names (``W_q``, ``W_k``, …) so that previously-saved models
        can still be loaded.  For ``num_blocks>1`` indexed keys are used
        (``W_q_0``, ``W_q_1``, …).
        """
        p = {
            'vocab_size': self.vocab_size, 'seq_len': self.seq_len,
            'embed_dim': self.embed_dim, 'num_classes': self.num_classes,
            'ff_dim': self.ff_dim, 'num_blocks': self.num_blocks,
            'embedding': self.embedding.copy(),
            'W_cls': self.W_cls.copy(), 'b_cls': self.b_cls.copy(),
        }
        if self.num_blocks == 1:
            # Backward-compatible flat keys
            p['W_q'] = self._W_q[0].copy()
            p['W_k'] = self._W_k[0].copy()
            p['W_v'] = self._W_v[0].copy()
            p['gamma_attn'] = self._gamma_attn[0].copy()
            p['beta_attn'] = self._beta_attn[0].copy()
            p['gamma_ff'] = self._gamma_ff[0].copy()
            p['beta_ff'] = self._beta_ff[0].copy()
            p['W_ff1'] = self._W_ff1[0].copy()
            p['b_ff1'] = self._b_ff1[0].copy()
            p['W_ff2'] = self._W_ff2[0].copy()
            p['b_ff2'] = self._b_ff2[0].copy()
        else:
            # Indexed keys for multi-block
            for i in range(self.num_blocks):
                p[f'W_q_{i}'] = self._W_q[i].copy()
                p[f'W_k_{i}'] = self._W_k[i].copy()
                p[f'W_v_{i}'] = self._W_v[i].copy()
                p[f'gamma_attn_{i}'] = self._gamma_attn[i].copy()
                p[f'beta_attn_{i}'] = self._beta_attn[i].copy()
                p[f'gamma_ff_{i}'] = self._gamma_ff[i].copy()
                p[f'beta_ff_{i}'] = self._beta_ff[i].copy()
                p[f'W_ff1_{i}'] = self._W_ff1[i].copy()
                p[f'b_ff1_{i}'] = self._b_ff1[i].copy()
                p[f'W_ff2_{i}'] = self._W_ff2[i].copy()
                p[f'b_ff2_{i}'] = self._b_ff2[i].copy()
        return p

    @classmethod
    def from_params(cls, params):
        """Reconstruct a model from a parameter dict (produced by ``get_params``)."""
        num_blocks = params.get('num_blocks', 1)
        model = cls(params['vocab_size'], params['seq_len'], params['embed_dim'],
                    params['num_classes'], ff_dim=params.get('ff_dim'),
                    num_blocks=num_blocks, seed=0)
        model.embedding = params['embedding']
        model.W_cls = params['W_cls']
        model.b_cls = params['b_cls']

        if num_blocks == 1:
            for key in cls._BLOCK_PARAM_NAMES:
                setattr(model, key, params[key])
            model._W_q[0] = params['W_q']
            model._W_k[0] = params['W_k']
            model._W_v[0] = params['W_v']
            model._gamma_attn[0] = params['gamma_attn']
            model._beta_attn[0] = params['beta_attn']
            model._gamma_ff[0] = params['gamma_ff']
            model._beta_ff[0] = params['beta_ff']
            model._W_ff1[0] = params['W_ff1']
            model._b_ff1[0] = params['b_ff1']
            model._W_ff2[0] = params['W_ff2']
            model._b_ff2[0] = params['b_ff2']
        else:
            for i in range(num_blocks):
                model._W_q[i] = params[f'W_q_{i}']
                model._W_k[i] = params[f'W_k_{i}']
                model._W_v[i] = params[f'W_v_{i}']
                model._gamma_attn[i] = params[f'gamma_attn_{i}']
                model._beta_attn[i] = params[f'beta_attn_{i}']
                model._gamma_ff[i] = params[f'gamma_ff_{i}']
                model._beta_ff[i] = params[f'beta_ff_{i}']
                model._W_ff1[i] = params[f'W_ff1_{i}']
                model._b_ff1[i] = params[f'b_ff1_{i}']
                model._W_ff2[i] = params[f'W_ff2_{i}']
                model._b_ff2[i] = params[f'b_ff2_{i}']
        return model

    def save(self, filename):
        params = self.get_params()
        header, tensors = serialize_model_dict(params)
        save_model(filename, header, tensors)

    @classmethod
    def load(cls, filename):
        from ..ia_format import load_model as _lm
        header, tensors = _lm(filename)
        d = dict(header)
        for k in header.get('_tensors', []):
            d[k] = tensors[k]
        d.pop('_tensors', None)
        return cls.from_params(d)


# ==================================================================
# Données par défaut
# ==================================================================

DEFAULT_VOCAB = {
    "bonjour": 0, "le": 1, "chat": 2, "mange": 3, "la": 4,
    "souris": 5, "petite": 6, "dort": 7, "sur": 8, "PAD": 9,
}

DEFAULT_NEXT_WORD_EXAMPLES = [
    ([0, 1, 2, 9], 3),   # "bonjour le chat" -> "mange"
    ([1, 2, 3, 9], 4),   # "le chat mange" -> "la"
    ([2, 3, 4, 9], 5),   # "chat mange la" -> "souris"
    ([3, 4, 5, 9], 6),   # "mange la souris" -> "petite"
    ([4, 5, 6, 9], 7),   # "la souris petite" -> "dort"
    ([5, 6, 7, 9], 8),   # "souris petite dort" -> "sur"
    ([6, 7, 8, 9], 1),   # "petite dort sur" -> "le"
    ([7, 8, 1, 9], 2),   # "dort sur le" -> "chat"
    ([8, 1, 2, 9], 3),   # "sur le chat" -> "mange"
    ([0, 1, 2, 3], 9),   # "bonjour le chat mange" -> PAD
    ([1, 2, 3, 4], 9),   # "le chat mange la" -> PAD
    ([2, 3, 4, 5], 9),   # "chat mange la souris" -> PAD
    ([3, 4, 5, 6], 9),   # "mange la souris petite" -> PAD
    ([4, 5, 6, 7], 9),   # "la souris petite dort" -> PAD
    ([0, 1, 6, 9], 2),   # "bonjour le petite" -> "chat"
    ([0, 2, 8, 9], 5),   # "bonjour chat sur" -> "souris"
    ([1, 4, 2, 9], 3),   # "le la chat" -> "mange"
    ([4, 6, 5, 9], 7),   # "la petite souris" -> "dort"
    ([2, 6, 7, 9], 8),   # "chat petite dort" -> "sur"
    ([1, 2, 7, 9], 3),   # "le chat dort" -> "mange"
]

EMOTION_CLASSES = ["joie", "tristesse", "colere", "peur", "surprise", "neutre"]
EMOTION_SENTENCES = [
    "je suis tellement heureux",
    "quelle belle journee",
    "c est vraiment merveilleux",
    "je ris de joie",
    "la vie est si belle",
    "je suis ravi de te voir",
    "je suis triste et melancolique",
    "c est tellement deccevant",
    "je pleure tout le temps",
    "tout va mal dans ma vie",
    "je me sens seul et abandonne",
    "c est un moment tres sombre",
    "je suis en colere contre lui",
    "ca m enerve profondement",
    "je suis furieux et irrite",
    "c est completement inacceptable",
    "je deteste cette situation",
    "ca me met hors de moi",
    "j ai tres peur du noir",
    "c est vraiment effrayant",
    "je suis terrorise par ca",
    "ca me fait peur la nuit",
    "je tremble de peur",
    "c est une situation angoissante",
    "quelle surprise incroyable",
    "je n y croyais pas du tout",
    "c est vraiment incroyable",
    "je suis completement etonne",
    "c est tellement surprenant",
    "je ne m y attendais pas",
    "c est une situation normale",
    "je ne sais pas quoi dire",
    "il n y a rien de special",
    "c est correct comme ca",
    "ca va bien merci",
    "pas de commentaire particulier",
]
EMOTION_LABELS = [0]*6 + [1]*6 + [2]*6 + [3]*6 + [4]*6 + [5]*6

MOOD_CLASSES = ["joyeux", "triste", "energique", "calme", "stresse", "fatigue", "motivé", "anxieux"]
MOOD_SENTENCES = [
    "je me sens joyeux et bien",
    "c est une journee joyeuse",
    "je suis plein de joie",
    "la vie me rend joyeux",
    "ce moment est joyeux",
    "je me sens triste aujourd hui",
    "une tristesse m envahit",
    "je suis triste et abattu",
    "c est un jour triste",
    "la melancolie me gagne",
    "je suis plein d energie",
    "quel dynamisme incroyable",
    "je me sens energique et fort",
    "mon energie est au maximum",
    "je deborde d energie",
    "je me sens calme et serein",
    "la paix interieure me guide",
    "je suis completement calme",
    "ce silence est apaisant",
    "je reste calme en toute situation",
    "je suis tres stresse en ce moment",
    "le stress me submerge",
    "c est une situation stressante",
    "la pression me stresse",
    "je ne supporte plus ce stress",
    "je suis epuise et fatigue",
    "la fatigue me gagne",
    "je n ai plus d energie",
    "c est un etat de fatigue totale",
    "je suis mort de fatigue",
    "je suis tres motive pour avancer",
    "ma motivation est intacte",
    "je reste motive malgre tout",
    "cette motivation me pousse",
    "je suis determine et motive",
    "je me sens anxieux sans raison",
    "l anxiete me ronge",
    "c est angoissant et anxieux",
    "je suis en proie a l anxiete",
    "mon angoisse est forte",
]
MOOD_LABELS = [0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5 + [6]*5 + [7]*5

STATEMENT_CLASSES = ["question", "affirmation", "ordre", "conseil", "exclamation"]
STATEMENT_SENTENCES = [
    "comment vas tu aujourd hui",
    "pourquoi tu fais ca",
    "qu est ce que tu veux",
    "ou est ce que tu vas",
    "quand est ce que tu arrives",
    "qui est cette personne",
    "combien ca coute",
    "est ce que tu comprends",
    "pourquoi pas",
    "a quelle heure viens tu",
    "je suis content de te voir",
    "le ciel est bleu aujourd hui",
    "c est une belle journee",
    "je travaille tous les jours",
    "la musique est douce",
    "ce plat est delicieux",
    "je vis dans une grande ville",
    "il fait beau ce matin",
    "le livre est sur la table",
    "nous avons fini le projet",
    "ferme la porte immediatement",
    "viens ici tout de suite",
    "arrete de parler",
    "mets tes chaussures",
    "range ta chambre",
    "ecoute moi bien",
    "ne bouge pas",
    "donne moi ce livre",
    "assieds toi",
    "partons maintenant",
    "tu devrais faire du sport",
    "il vaut mieux etre prudent",
    "je te conseille de lire",
    "essaie de te reposer",
    "pense a boire de l eau",
    "il faudrait partir tot",
    "tu devrais dormir plus",
    "mieux vaut prevoir large",
    "je suggere de prendre un cafe",
    "quel temps magnifique",
    "bravo c est superbe",
    "quelle horreur cette situation",
    "incroyable mais vrai",
    "oh la la quelle surprise",
    "c est formidable",
    "je n en reviens pas",
    "quelle chance incroyable",
    "magnifique vraiment",
]
STATEMENT_LABELS = [0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10

SENTIMENT_CLASSES = ["positif", "negatif", "neutre"]
SENTIMENT_SENTENCES = [
    "c est un excellent travail",
    "je suis tres satisfait du resultat",
    "merci beaucoup pour ton aide",
    "c est une belle reussite",
    "je recommande vivement ce film",
    "quelle bonne nouvelle",
    "tout s est bien passe",
    "c est absolument genial",
    "je suis ravi de cette nouvelle",
    "bravo pour cette performance",
    "c est terrible et affreux",
    "je deteste cette situation",
    "c est le pire moment de ma vie",
    "je suis decu et furieux",
    "c est completement nul",
    "quelle perte de temps",
    "je regrette amrement ce choix",
    "c est une catastrophe totale",
    "je suis malheureux et triste",
    "ce n est vraiment pas bien",
    "le train arrive a dix heures",
    "il y a trois personnes dans la piece",
    "le document contient cinq pages",
    "c est un mardi de novembre",
    "la reunion est prevue ce matin",
    "le magasin est ferme le lundi",
    "il pleut depuis ce matin",
    "le livre a ete publie en mars",
    "je porte une chemise bleue",
    "la voiture est garée devant",
]
SENTIMENT_LABELS = [0]*10 + [1]*10 + [2]*10


# ==================================================================
# Fonctions d'entrainement SLM
# ==================================================================

def _train_slm_core(model, X_train, y_train, pad_masks, epochs, lr,
                    save_path, class_names, preprocessor_params=None,
                    extra_meta=None):
    """Boucle d'entrainement commune pour tous les modeles SLM de classification."""

    best_loss = float('inf')
    best_params = None
    history = []

    for epoch in range(epochs):
        total_loss = 0
        indices = C.permutation(len(X_train), _next_seed())
        for idx in indices:
            x = X_train[idx]
            y = y_train[idx]
            mask = pad_masks[idx] if pad_masks is not None else None
            out = model.forward(x, pad_mask=mask)
            loss = -math.log(float(out[y]) + 1e-8)
            total_loss += loss
            model.backward(x, y, out, lr)

        avg_loss = total_loss / len(X_train)
        history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = model.get_params()

        if epoch % 100 == 0:
            logger.info("Epoch %d, Loss: %.6f", epoch, avg_loss)
        if avg_loss < 0.01:
            logger.info("Convergence a l'epoch %d", epoch)
            break

    # Restaurer le meilleur modele et evaluer
    best_model = SLMClassifier.from_params(best_params)
    correct = 0
    for i in range(len(X_train)):
        mask = pad_masks[i] if pad_masks is not None else None
        out = best_model.forward(X_train[i], pad_mask=mask)
        pred = C.argmax(out)
        if pred == y_train[i]:
            correct += 1
    accuracy = correct / len(X_train) * 100

    model_dict = {
        'network_params': best_params,
        'class_names': class_names,
        'accuracy': accuracy,
        'seq_len': best_model.seq_len,
        'embed_dim': best_model.embed_dim,
        'ff_dim': best_model.ff_dim,
        'num_classes': best_model.num_classes,
        'vocab_size': best_model.vocab_size,
        'num_blocks': best_model.num_blocks,
    }
    if preprocessor_params is not None:
        model_dict['preprocessor_params'] = preprocessor_params
    if extra_meta is not None:
        model_dict.update(extra_meta)

    header, tensors = serialize_model_dict(model_dict)
    save_model(save_path, header, tensors)
    logger.info("SLM sauvegarde dans %s (precision: %.1f%%)", save_path, accuracy)

    return {'model': model_dict, 'save_path': save_path, 'accuracy': accuracy, 'history': history}


def train_slm_next_word(vocab=None, seq_len=4, embed_dim=16, ff_dim=32,
                        lr=0.01, epochs=1000, save_path=None, seed=42,
                        num_blocks=1):
    """
    Entraîne un SLM pour la prédiction du mot suivant.

    Args:
        vocab: Dictionnaire mot -> index (défaut : 10 mots français).
        seq_len: Longueur des séquences d'entrée.
        embed_dim: Dimension d'embedding.
        ff_dim: Dimension du feed-forward.
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        num_blocks: Nombre de blocs transformer (défaut 1, backward compatible).

    Returns:
        dict: Modèle et métriques.
    """
    ensure_directories()

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"slm_next_word{MODEL_EXTENSION}")

    if vocab is None:
        vocab = DEFAULT_VOCAB

    vocab_size = len(vocab)
    pad_idx = vocab.get("PAD", vocab_size - 1)

    examples = DEFAULT_NEXT_WORD_EXAMPLES
    X_train = np.array([np.array(e[0]) for e in examples])
    y_train = np.array([e[1] for e in examples])
    pad_masks = np.array([[t != pad_idx for t in e[0]] for e in examples])

    model = SLMClassifier(vocab_size, seq_len, embed_dim, vocab_size,
                          ff_dim=ff_dim, num_blocks=num_blocks, seed=seed)

    result = _train_slm_core(
        model, X_train, y_train, pad_masks, epochs, lr,
        save_path, list(vocab.keys()),
        extra_meta={'vocab': vocab},
    )
    return result


def train_slm_emotion(sentences=None, labels=None, seq_len=10, embed_dim=16,
                      ff_dim=32, lr=0.01, epochs=500, save_path=None, seed=42,
                      num_blocks=1):
    """
    Entraîne un SLM pour la détection d'émotion.

    6 classes : joie, tristesse, colère, peur, surprise, neutre.

    Args:
        sentences: Liste de phrases (défaut : 36 phrases françaises).
        labels: Liste d'indices de classe.
        seq_len: Longueur des séquences.
        embed_dim: Dimension d'embedding.
        ff_dim: Dimension du feed-forward.
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        num_blocks: Nombre de blocs transformer (défaut 1, backward compatible).

    Returns:
        dict: Modèle et métriques.
    """
    ensure_directories()

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"slm_emotion{MODEL_EXTENSION}")

    if sentences is None or labels is None:
        sentences = EMOTION_SENTENCES
        labels = EMOTION_LABELS

    num_classes = len(EMOTION_CLASSES)
    preprocessor = TextPreprocessor()
    preprocessor.fit(sentences)

    X_train = np.array([preprocessor.encode(s, seq_len) for s in sentences])
    y_train = np.array(labels)

    model = SLMClassifier(preprocessor.vocab_size, seq_len, embed_dim,
                          num_classes, ff_dim=ff_dim, num_blocks=num_blocks,
                          seed=seed)

    return _train_slm_core(
        model, X_train, y_train, None, epochs, lr,
        save_path, EMOTION_CLASSES,
        preprocessor_params=preprocessor.get_params(),
    )


def train_slm_mood(sentences=None, labels=None, seq_len=10, embed_dim=16,
                   ff_dim=32, lr=0.01, epochs=500, save_path=None, seed=42,
                   num_blocks=1):
    """
    Entraîne un SLM pour la détection d'humeur.

    8 classes : joyeux, triste, énergique, calmé, stressé, fatigué, motivé, anxieux.

    Args:
        sentences: Liste de phrases (défaut : 40 phrases françaises).
        labels: Liste d'indices de classe.
        seq_len: Longueur des séquences.
        embed_dim: Dimension d'embedding.
        ff_dim: Dimension du feed-forward.
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        num_blocks: Nombre de blocs transformer (défaut 1, backward compatible).

    Returns:
        dict: Modèle et métriques.
    """
    ensure_directories()

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"slm_mood{MODEL_EXTENSION}")

    if sentences is None or labels is None:
        sentences = MOOD_SENTENCES
        labels = MOOD_LABELS

    num_classes = len(MOOD_CLASSES)
    preprocessor = TextPreprocessor()
    preprocessor.fit(sentences)

    X_train = np.array([preprocessor.encode(s, seq_len) for s in sentences])
    y_train = np.array(labels)

    model = SLMClassifier(preprocessor.vocab_size, seq_len, embed_dim,
                          num_classes, ff_dim=ff_dim, num_blocks=num_blocks,
                          seed=seed)

    return _train_slm_core(
        model, X_train, y_train, None, epochs, lr,
        save_path, MOOD_CLASSES,
        preprocessor_params=preprocessor.get_params(),
    )


def train_slm_statement(sentences=None, labels=None, seq_len=10, embed_dim=16,
                        ff_dim=32, lr=0.01, epochs=500, save_path=None, seed=42,
                        num_blocks=1):
    """
    Entraîne un SLM pour la classification de type de phrase.

    5 classes : question, affirmation, ordre, conseil, exclamation.

    Args:
        sentences: Liste de phrases (défaut : 50 phrases françaises).
        labels: Liste d'indices de classe.
        seq_len: Longueur des séquences.
        embed_dim: Dimension d'embedding.
        ff_dim: Dimension du feed-forward.
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        num_blocks: Nombre de blocs transformer (défaut 1, backward compatible).

    Returns:
        dict: Modèle et métriques.
    """
    ensure_directories()

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"slm_statement{MODEL_EXTENSION}")

    if sentences is None or labels is None:
        sentences = STATEMENT_SENTENCES
        labels = STATEMENT_LABELS

    num_classes = len(STATEMENT_CLASSES)
    preprocessor = TextPreprocessor()
    preprocessor.fit(sentences)

    X_train = np.array([preprocessor.encode(s, seq_len) for s in sentences])
    y_train = np.array(labels)

    model = SLMClassifier(preprocessor.vocab_size, seq_len, embed_dim,
                          num_classes, ff_dim=ff_dim, num_blocks=num_blocks,
                          seed=seed)

    return _train_slm_core(
        model, X_train, y_train, None, epochs, lr,
        save_path, STATEMENT_CLASSES,
        preprocessor_params=preprocessor.get_params(),
    )


def train_slm_sentiment(sentences=None, labels=None, seq_len=10, embed_dim=16,
                        ff_dim=32, lr=0.01, epochs=500, save_path=None, seed=42,
                        num_blocks=1):
    """
    Entraîne un SLM pour l'analyse de sentiment.

    3 classes : positif, négatif, neutre.

    Args:
        sentences: Liste de phrases (défaut : 30 phrases françaises).
        labels: Liste d'indices de classe.
        seq_len: Longueur des séquences.
        embed_dim: Dimension d'embedding.
        ff_dim: Dimension du feed-forward.
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        num_blocks: Nombre de blocs transformer (défaut 1, backward compatible).

    Returns:
        dict: Modèle et métriques.
    """
    ensure_directories()

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"slm_sentiment{MODEL_EXTENSION}")

    if sentences is None or labels is None:
        sentences = SENTIMENT_SENTENCES
        labels = SENTIMENT_LABELS

    num_classes = len(SENTIMENT_CLASSES)
    preprocessor = TextPreprocessor()
    preprocessor.fit(sentences)

    X_train = np.array([preprocessor.encode(s, seq_len) for s in sentences])
    y_train = np.array(labels)

    model = SLMClassifier(preprocessor.vocab_size, seq_len, embed_dim,
                          num_classes, ff_dim=ff_dim, num_blocks=num_blocks,
                          seed=seed)

    return _train_slm_core(
        model, X_train, y_train, None, epochs, lr,
        save_path, SENTIMENT_CLASSES,
        preprocessor_params=preprocessor.get_params(),
    )