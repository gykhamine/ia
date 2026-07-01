"""
IA/train/transformer.py - Entraînement des Transformers.

Modèles :
  - MiniTransformer : single-head, classification de séquences token.
    Supporte un nombre configurable de blocs transformer (num_blocks).
  - MiniTransformer3D : multi-head batched, classification de séquences.
"""

import math
import numpy as np
import pickle
import os
import logging

from ..config import MODELS_DIR, MODEL_EXTENSION, ensure_directories
from ..cpp import get_core

C = get_core()

logger = logging.getLogger(__name__)

_seed_state = 42


def _next_seed():
    global _seed_state
    s = _seed_state
    _seed_state += 1
    return s


# ==================================================================
# Fonctions communes
# ==================================================================

def softmax(x):
    return C.softmax(x)


def relu(x):
    return C.relu(x)


def relu_deriv(x):
    return C.relu_deriv(x)


def layer_norm(x, eps=1e-8):
    return C.layer_norm(x, eps)


def xavier_init(shape):
    return C.xavier_init(tuple(shape), _next_seed())


# ==================================================================
# MiniTransformer (single-head)
# ==================================================================

def train_transformer(X=None, y=None, seq_len=4, vocab_size=6,
                       embed_dim=8, ff_dim=16, lr=0.01, epochs=2000,
                       early_stop_loss=0.001, save_path=None, seed=42,
                       num_blocks=1):
    """
    Entraîne un Mini-Transformer single-head pour la classification de séquences.

    Args:
        X: Tokens d'entrée (batch, seq_len).
        y: Labels (batch, 1).
        seq_len: Longueur des séquences.
        vocab_size: Taille du vocabulaire.
        embed_dim: Dimension d'embedding.
        ff_dim: Dimension du feed-forward.
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        early_stop_loss: Seuil de convergence.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        num_blocks: Nombre de blocs transformer (défaut 1). Chaque bloc
            contient une sous-couche d'attention et une sous-couche
            feed-forward avec connexions résiduelles et normalisation de couche.
            Quand num_blocks=1, le comportement est identique à la version
            originale (rétrocompatible).

    Returns:
        dict: Modèle et métriques.
    """
    ensure_directories()
    global _seed_state
    _seed_state = seed

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"mini_transformer{MODEL_EXTENSION}")

    if X is None or y is None:
        X = np.array([
            [1, 2, 3, 4], [4, 3, 2, 1], [2, 2, 2, 2], [1, 3, 1, 3],
            [0, 1, 2, 3], [5, 4, 3, 2], [1, 1, 2, 2], [3, 2, 3, 2],
        ])
        y = np.array([[1], [0], [1], [0], [1], [0], [1], [0]])

    # Initialisation
    embedding = xavier_init((vocab_size, embed_dim))
    W_qs = [xavier_init((embed_dim, embed_dim)) for _ in range(num_blocks)]
    W_ks = [xavier_init((embed_dim, embed_dim)) for _ in range(num_blocks)]
    W_vs = [xavier_init((embed_dim, embed_dim)) for _ in range(num_blocks)]
    gamma_attns = [np.ones((1, embed_dim)) for _ in range(num_blocks)]
    beta_attns = [np.zeros((1, embed_dim)) for _ in range(num_blocks)]
    gamma_ffs = [np.ones((1, embed_dim)) for _ in range(num_blocks)]
    beta_ffs = [np.zeros((1, embed_dim)) for _ in range(num_blocks)]
    W_ff1s = [xavier_init((embed_dim, ff_dim)) for _ in range(num_blocks)]
    b_ff1s = [np.zeros((1, ff_dim)) for _ in range(num_blocks)]
    W_ff2s = [xavier_init((ff_dim, embed_dim)) for _ in range(num_blocks)]
    b_ff2s = [np.zeros((1, embed_dim)) for _ in range(num_blocks)]
    W_cls = xavier_init((embed_dim, 1))
    b_cls = np.zeros((1, 1))

    best_loss = float('inf')
    best_params = None
    history = []

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            # ---- Forward ----
            x_emb = embedding[X[i]]
            x = x_emb
            block_caches = []

            # Pré-allouer les listes de gradients par bloc (zéros) — évite
            # NameError quand on écrit `dW_qs[b_idx] = ...` plus bas.
            # Indispensable pour num_blocks > 1 ; sans danger pour num_blocks = 1.
            dW_qs = [C.zeros(W_qs[b].shape) for b in range(num_blocks)]
            dW_ks = [C.zeros(W_ks[b].shape) for b in range(num_blocks)]
            dW_vs = [C.zeros(W_vs[b].shape) for b in range(num_blocks)]
            dW_ff1s = [C.zeros(W_ff1s[b].shape) for b in range(num_blocks)]
            db_ff1s = [C.zeros(b_ff1s[b].shape) for b in range(num_blocks)]
            dW_ff2s = [C.zeros(W_ff2s[b].shape) for b in range(num_blocks)]
            db_ff2s = [C.zeros(b_ff2s[b].shape) for b in range(num_blocks)]

            for b_idx in range(num_blocks):
                bc = {}
                bc['x_in'] = x
                Q = C.matmul(x, W_qs[b_idx]); K = C.matmul(x, W_ks[b_idx]); V = C.matmul(x, W_vs[b_idx])
                bc['Q'] = Q; bc['K'] = K; bc['V'] = V
                attn_scores = C.matmul(Q, K.T) / math.sqrt(embed_dim)
                attn_weights = softmax(attn_scores)
                bc['attn_weights'] = attn_weights
                attn_output = C.matmul(attn_weights, V)
                bc['attn_output'] = attn_output
                x = x + attn_output
                x = layer_norm(x) * gamma_attns[b_idx] + beta_attns[b_idx]
                bc['x_attn_norm'] = x
                ff1 = relu(C.matmul(x, W_ff1s[b_idx]) + b_ff1s[b_idx])
                bc['ff1'] = ff1
                ff2 = C.matmul(ff1, W_ff2s[b_idx]) + b_ff2s[b_idx]
                bc['ff2'] = ff2
                x = x + ff2
                x = layer_norm(x) * gamma_ffs[b_idx] + beta_ffs[b_idx]
                block_caches.append(bc)

            pooled = C.mean_axis(x, 0).reshape(1, -1)
            logits = C.matmul(pooled, W_cls) + b_cls
            out = C.sigmoid(logits)

            error = y[i] - out
            loss = C.sum(error ** 2)
            total_loss += loss

            # ---- Backward - classification head ----
            d_logits = error * out * (1 - out)
            d_pooled = C.matmul(d_logits, W_cls.T)
            dW_cls = C.matmul(pooled.T, d_logits)
            db_cls = d_logits

            # Backward - mean pooling
            d_x = C.tile(d_pooled / seq_len, seq_len)

            # ---- Backward - blocks in reverse order ----
            for b_idx in range(num_blocks - 1, -1, -1):
                bc = block_caches[b_idx]
                x_in = bc['x_in']
                Q = bc['Q']; K = bc['K']; V = bc['V']
                attn_weights = bc['attn_weights']
                attn_output = bc['attn_output']
                x_attn_norm = bc['x_attn_norm']
                ff1 = bc['ff1']
                ff2 = bc['ff2']

                # FFN sub-block backward
                d_ff2 = d_x
                d_x_attn_norm = d_x
                d_ff1 = C.matmul(d_ff2, W_ff2s[b_idx].T) * relu_deriv(ff1)
                dW_ff2s[b_idx] = C.matmul(ff1.T, d_ff2)
                db_ff2s[b_idx] = C.sum_axis(d_ff2, 0).reshape(1, -1)
                dW_ff1s[b_idx] = C.matmul(x_attn_norm.T, d_ff1)
                db_ff1s[b_idx] = C.sum_axis(d_ff1, 0).reshape(1, -1)

                # Attention sub-block backward
                d_x_attn = d_x_attn_norm
                d_x_in_attn = d_x_attn
                d_attn_output = d_x_attn
                d_attn_weights = C.matmul(d_attn_output, V.T)
                d_V = C.matmul(attn_weights.T, d_attn_output)
                d_attn_scores = (attn_weights * (d_attn_weights -
                                  C.sum_axis(attn_weights * d_attn_weights, -1).reshape(seq_len, 1)))
                d_attn_scores = d_attn_scores / math.sqrt(embed_dim)
                d_Q = C.matmul(d_attn_scores, K); d_K = C.matmul(d_attn_scores.T, Q)
                d_x_in_q = C.matmul(d_Q, W_qs[b_idx].T); d_x_in_k = C.matmul(d_K, W_ks[b_idx].T)
                d_x_in_v = C.matmul(d_V, W_vs[b_idx].T)
                dW_qs[b_idx] = C.matmul(x_in.T, d_Q)
                dW_ks[b_idx] = C.matmul(x_in.T, d_K)
                dW_vs[b_idx] = C.matmul(x_in.T, d_V)

                # Gradient flowing to input of this block
                d_x = d_x_in_q + d_x_in_k + d_x_in_v + d_x_in_attn

            # d_x is now the gradient w.r.t. x_emb
            W_cls += lr * dW_cls; b_cls += lr * db_cls
            for b_idx in range(num_blocks):
                W_ff1s[b_idx] += lr * dW_ff1s[b_idx]; b_ff1s[b_idx] += lr * db_ff1s[b_idx]
                W_ff2s[b_idx] += lr * dW_ff2s[b_idx]; b_ff2s[b_idx] += lr * db_ff2s[b_idx]
                W_qs[b_idx] += lr * dW_qs[b_idx]; W_ks[b_idx] += lr * dW_ks[b_idx]; W_vs[b_idx] += lr * dW_vs[b_idx]

            for j, token_id in enumerate(X[i]):
                embedding[token_id] += lr * d_x[j]

            W_cls = C.clip(W_cls, -1, 1)
            for b_idx in range(num_blocks):
                W_qs[b_idx] = C.clip(W_qs[b_idx], -1, 1)
                W_ks[b_idx] = C.clip(W_ks[b_idx], -1, 1)
                W_vs[b_idx] = C.clip(W_vs[b_idx], -1, 1)
                W_ff1s[b_idx] = C.clip(W_ff1s[b_idx], -1, 1)
                W_ff2s[b_idx] = C.clip(W_ff2s[b_idx], -1, 1)

        avg_loss = total_loss / len(X)
        history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            if num_blocks == 1:
                best_params = {
                    'embedding': embedding.copy(), 'W_q': W_qs[0].copy(), 'W_k': W_ks[0].copy(),
                    'W_v': W_vs[0].copy(), 'W_ff1': W_ff1s[0].copy(), 'b_ff1': b_ff1s[0].copy(),
                    'W_ff2': W_ff2s[0].copy(), 'b_ff2': b_ff2s[0].copy(),
                    'W_cls': W_cls.copy(), 'b_cls': b_cls.copy(),
                }
            else:
                best_params = {
                    'embedding': embedding.copy(),
                    'W_q_list': [w.copy() for w in W_qs],
                    'W_k_list': [w.copy() for w in W_ks],
                    'W_v_list': [w.copy() for w in W_vs],
                    'W_ff1_list': [w.copy() for w in W_ff1s],
                    'b_ff1_list': [w.copy() for w in b_ff1s],
                    'W_ff2_list': [w.copy() for w in W_ff2s],
                    'b_ff2_list': [w.copy() for w in b_ff2s],
                    'W_cls': W_cls.copy(), 'b_cls': b_cls.copy(),
                    'num_blocks': num_blocks,
                }

        if epoch % 200 == 0:
            logger.info("Epoch %d, Loss: %.6f", epoch, avg_loss)
        if avg_loss < early_stop_loss:
            logger.info("Convergence à l'epoch %d", epoch)
            break

    # Charger le meilleur et évaluer
    p = best_params
    num_blocks_eval = p.get('num_blocks', 1)
    correct = 0
    for i in range(len(X)):
        x = p['embedding'][X[i]]
        for b_idx in range(num_blocks_eval):
            if num_blocks_eval == 1:
                wq, wk, wv = p['W_q'], p['W_k'], p['W_v']
                wff1, bff1 = p['W_ff1'], p['b_ff1']
                wff2, bff2 = p['W_ff2'], p['b_ff2']
            else:
                wq, wk, wv = p['W_q_list'][b_idx], p['W_k_list'][b_idx], p['W_v_list'][b_idx]
                wff1, bff1 = p['W_ff1_list'][b_idx], p['b_ff1_list'][b_idx]
                wff2, bff2 = p['W_ff2_list'][b_idx], p['b_ff2_list'][b_idx]
            Q = C.matmul(x, wq); K = C.matmul(x, wk); V = C.matmul(x, wv)
            attn_scores = C.matmul(Q, K.T) / math.sqrt(embed_dim)
            attn_weights = softmax(attn_scores)
            attn_output = C.matmul(attn_weights, V)
            x = x + attn_output
            x = layer_norm(x)
            ff1 = relu(C.matmul(x, wff1) + bff1)
            ff2 = C.matmul(ff1, wff2) + bff2
            x = x + ff2
            x = layer_norm(x)
        pooled = C.mean_axis(x, 0).reshape(1, -1)
        out = C.sigmoid(C.matmul(pooled, p['W_cls']) + p['b_cls'])
        pred = 1 if out[0, 0] > 0.5 else 0
        if pred == y[i][0]:
            correct += 1
    accuracy = correct / len(X) * 100

    model = {**p, 'seq_len': seq_len, 'vocab_size': vocab_size,
             'embed_dim': embed_dim, 'accuracy': accuracy}
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info("Transformer sauvegardé dans %s (précision: %.1f%%)", save_path, accuracy)

    return {'model': model, 'save_path': save_path, 'accuracy': accuracy, 'history': history}


# ==================================================================
# MiniTransformer3D (multi-head batched)
# ==================================================================

class MiniTransformer3D:
    """Transformer 3D multi-head avec batch."""

    def __init__(self, vocab_size, seq_len, embed_dim, ff_dim, num_heads=2, seed=42):
        global _seed_state
        _seed_state = seed
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.embedding = xavier_init((vocab_size, embed_dim))
        self.W_q = xavier_init((num_heads, embed_dim, self.head_dim))
        self.W_k = xavier_init((num_heads, embed_dim, self.head_dim))
        self.W_v = xavier_init((num_heads, embed_dim, self.head_dim))
        self.W_o = xavier_init((num_heads * self.head_dim, embed_dim))
        self.gamma_attn = np.ones((1, embed_dim))
        self.beta_attn = np.zeros((1, embed_dim))
        self.gamma_ff = np.ones((1, embed_dim))
        self.beta_ff = np.zeros((1, embed_dim))
        self.W_ff1 = xavier_init((embed_dim, ff_dim))
        self.b_ff1 = np.zeros((1, ff_dim))
        self.W_ff2 = xavier_init((ff_dim, embed_dim))
        self.b_ff2 = np.zeros((1, embed_dim))
        self.W_cls = xavier_init((embed_dim, 1))
        self.b_cls = np.zeros((1, 1))

    def forward(self, x):
        """Forward pass. x: (batch_size, seq_len)."""
        batch_size, seq_len = x.shape
        x_emb = self.embedding[x]
        attn_output, attn_weights = self._multi_head_attention(x_emb)
        x_attn = x_emb + attn_output
        x_attn_norm = C.zeros(tuple(x_attn.shape))
        for b in range(batch_size):
            for pos in range(seq_len):
                x_attn_norm[b, pos] = layer_norm(x_attn[b, pos]) * self.gamma_attn[0] + self.beta_attn[0]
        ff1 = relu(C.matmul(x_attn_norm, self.W_ff1) + self.b_ff1)
        ff2 = C.matmul(ff1, self.W_ff2) + self.b_ff2
        x_ff = x_attn_norm + ff2
        x_ff_norm = C.zeros(tuple(x_ff.shape))
        for b in range(batch_size):
            for pos in range(seq_len):
                x_ff_norm[b, pos] = layer_norm(x_ff[b, pos]) * self.gamma_ff[0] + self.beta_ff[0]
        pooled = C.mean_axis(x_ff_norm, 1)
        logits = C.matmul(pooled, self.W_cls) + self.b_cls
        out = C.sigmoid(logits)
        cache = {
            'x_emb': x_emb, 'attn_output': attn_output, 'attn_weights': attn_weights,
            'x_attn_norm': x_attn_norm, 'ff1': ff1, 'ff2': ff2, 'x_ff_norm': x_ff_norm,
            'pooled': pooled, 'logits': logits,
        }
        return out, cache

    def _multi_head_attention(self, x):
        batch_size, seq_len, embed_dim = x.shape
        head_outputs = []
        attn_weights_list = []
        for head in range(self.num_heads):
            Q = C.matmul(x, self.W_q[head]); K = C.matmul(x, self.W_k[head]); V = C.matmul(x, self.W_v[head])
            attn_scores = C.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(self.head_dim)
            attn_weights = softmax(attn_scores)
            head_output = C.matmul(attn_weights, V)
            head_outputs.append(head_output)
            attn_weights_list.append(attn_weights)
        multi_head_output = C.concatenate(head_outputs, -1)
        output = C.matmul(multi_head_output, self.W_o)
        return output, attn_weights_list

    def backward(self, x, y, out, cache, lr=0.01):
        batch_size, seq_len = x.shape
        error = y - out
        d_logits = error * out * (1 - out)
        d_pooled = C.matmul(d_logits, self.W_cls.T)
        dW_cls = C.matmul(cache['pooled'].T, d_logits)
        db_cls = C.sum_axis(d_logits, 0).reshape(1, -1)
        d_x_ff_norm = C.tile(d_pooled[:, np.newaxis, :] / seq_len, (1, seq_len, 1))
        d_x_attn_norm = d_x_ff_norm
        d_ff2 = d_x_ff_norm
        d_ff1 = C.matmul(d_ff2, self.W_ff2.T) * relu_deriv(cache['ff1'])
        dW_ff2 = C.matmul(cache['ff1'].transpose(0, 2, 1), d_ff2)
        db_ff2 = C.sum_axis(C.sum_axis(d_ff2, 0), 0).reshape(1, -1)
        dW_ff1 = C.matmul(cache['x_attn_norm'].transpose(0, 2, 1), d_ff1)
        db_ff1 = C.sum_axis(C.sum_axis(d_ff1, 0), 0).reshape(1, -1)
        d_x_attn = d_x_attn_norm
        d_attn_output = d_x_attn
        d_W_o = C.zeros(tuple(self.W_o.shape))
        for head in range(self.num_heads):
            start = head * self.head_dim
            end = (head + 1) * self.head_dim
            d_head_output = C.matmul(d_attn_output, self.W_o[start:end, :].T)
            dW_q_h = C.mean_axis(C.matmul(cache['x_emb'].transpose(0, 2, 1), d_head_output), 0)
            dW_k_h = C.mean_axis(C.matmul(cache['x_emb'].transpose(0, 2, 1), d_head_output), 0)
            dW_v_h = C.mean_axis(C.matmul(cache['x_emb'].transpose(0, 2, 1), d_head_output), 0)
            self.W_q[head] += lr * dW_q_h
            self.W_k[head] += lr * dW_k_h
            self.W_v[head] += lr * dW_v_h
            d_W_o[start:end, :] += C.mean_axis(
                C.matmul(cache['attn_output'][:, :, start:end].transpose(0, 2, 1), d_attn_output), 0)
        self.W_o += lr * d_W_o
        d_x_emb_total = d_x_attn
        for b in range(batch_size):
            for pos in range(seq_len):
                token_id = x[b, pos]
                self.embedding[token_id] += lr * np.mean(d_x_emb_total[b, pos])
        self.W_cls += lr * dW_cls; self.b_cls += lr * db_cls
        self.W_ff1 += lr * C.mean_axis(dW_ff1, 0)
        self.W_ff2 += lr * C.mean_axis(dW_ff2, 0)
        self.b_ff1 += lr * C.sum(db_ff1) / db_ff1.size
        self.b_ff2 += lr * C.sum(db_ff2) / db_ff2.size
        for attr in ['W_q', 'W_k', 'W_v', 'W_o', 'W_ff1', 'W_ff2', 'W_cls']:
            setattr(self, attr, C.clip(getattr(self, attr), -1, 1))

    def save(self, filename):
        model_data = {k: getattr(self, k) for k in [
            'vocab_size', 'seq_len', 'embed_dim', 'ff_dim', 'num_heads', 'head_dim',
            'embedding', 'W_q', 'W_k', 'W_v', 'W_o',
            'gamma_attn', 'beta_attn', 'gamma_ff', 'beta_ff',
            'W_ff1', 'b_ff1', 'W_ff2', 'b_ff2', 'W_cls', 'b_cls']}
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            d = pickle.load(f)
        model = cls(d['vocab_size'], d['seq_len'], d['embed_dim'], d['ff_dim'], d['num_heads'])
        for k in ['embedding', 'W_q', 'W_k', 'W_v', 'W_o',
                   'gamma_attn', 'beta_attn', 'gamma_ff', 'beta_ff',
                   'W_ff1', 'b_ff1', 'W_ff2', 'b_ff2', 'W_cls', 'b_cls']:
            setattr(model, k, d[k])
        return model


def train_transformer3d(X=None, y=None, vocab_size=6, seq_len=4,
                         embed_dim=8, ff_dim=16, num_heads=2,
                         lr=0.01, epochs=5000, convergence_loss=0.01,
                         save_path=None, seed=42):
    """
    Entraîne un MiniTransformer3D multi-head batched.

    Args:
        X: Tokens (batch, seq_len).
        y: Labels (batch, 1).
        vocab_size, seq_len, embed_dim, ff_dim, num_heads: Hyperparamètres.
        lr, epochs: Optimisation.
        convergence_loss: Seuil de convergence.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.

    Returns:
        dict: Modèle et métriques.
    """
    ensure_directories()

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"transformer_3d{MODEL_EXTENSION}")

    if X is None or y is None:
        X = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 2, 2, 2], [1, 3, 1, 3]])
        y = np.array([[1], [0], [1], [0]])

    model = MiniTransformer3D(vocab_size, seq_len, embed_dim, ff_dim, num_heads, seed)
    history = []

    for epoch in range(epochs):
        out, cache = model.forward(X)
        loss = C.mse_loss(out, y)
        history.append(loss)
        model.backward(X, y, out, cache, lr)

        if epoch % 50 == 0:
            logger.info("Epoch %d, Loss: %.6f", epoch, loss)
        if loss < convergence_loss:
            logger.info("Convergence à l'epoch %d", epoch)
            break

    # Évaluation
    out, _ = model.forward(X)
    predictions = (out > 0.5).astype(int)
    accuracy = np.sum(predictions == y) / predictions.size * 100

    model.save(save_path)
    logger.info("Transformer3D sauvegardé dans %s (précision: %.1f%%)", save_path, accuracy)

    return {'model': model, 'save_path': save_path, 'accuracy': accuracy, 'history': history}