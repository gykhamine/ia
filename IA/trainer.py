"""
IA/trainer.py — Classe Trainer : point d'entrée unifié pour entraîner tous les modèles.

Usage :
    from IA import Trainer

    trainer = Trainer(verbose=True)
    model = trainer.train(type='rnn', epochs=500, lr=0.01)
    y_pred = model.predict(X_new)
    model.save('mon_rnn.gy')

    # Plus tard :
    model2 = Trainer.load('mon_rnn.gy')
    y_pred2 = model2.predict(X_new)

Architecture :
  Trainer.train() dispatche vers la fonction train_xxx correspondante,
  en injectant les callbacks à travers un wrapper qui convertit le
  format "history list" interne au format "callback events".

  Chaque train_xxx existant reste inchangé et peut être appelé directement
  pour rétro-compatibilité. Le Trainer ajoute par-dessus :
    - dispatch unifié par type
    - callbacks Keras-like (EarlyStopping, ModelCheckpoint, ...)
    - construction automatique d'un objet Model avec predict_fn enregistrée
    - sauvegarde .gy (binaire safe, seul format supporté)
"""
import math
import os
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from . import ia_format
from .callbacks import Callback, CallbackList, default_callbacks
from .model import Model, register_predict_fn, _PREDICT_REGISTRY
from .config import ensure_directories, MODELS_DIR, MODEL_EXTENSION
from .exceptions import (
    ConfigurationError, InferenceError, ModelFormatError, TrainingError,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Registre des types de modèles supportés
# ============================================================================
# Chaque entrée contient :
#   'train_fn'   : fonction train_xxx du paquet IA.train
#   'predict_fn' : fonction d'inférence (predict_fn(weights, config, X) -> y)
#   'default_kwargs' : kwargs additionnels à passer à train_fn
#   'extract_weights' : fonction (train_result) -> dict {nom: ndarray}
#   'extract_config'  : fonction (train_result) -> dict de config
#   'default_save_name' : nom de fichier par défaut (sans extension)

_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_model_type(model_type: str, train_fn: Callable,
                        predict_fn: Callable,
                        extract_weights: Callable,
                        extract_config: Callable,
                        default_save_name: str,
                        default_kwargs: Optional[Dict[str, Any]] = None):
    """Enregistre un type de modèle dans le registre du Trainer."""
    _REGISTRY[model_type] = {
        'train_fn': train_fn,
        'predict_fn': predict_fn,
        'extract_weights': extract_weights,
        'extract_config': extract_config,
        'default_save_name': default_save_name,
        'default_kwargs': default_kwargs or {},
    }
    # Enregistre aussi la predict_fn dans le registre de Model
    register_predict_fn(model_type)(predict_fn)


# ============================================================================
# Fonctions d'inférence pour chaque type de modèle
# ============================================================================
#Ces fonctions reproduisent le forward pass en utilisant les poids stockés
# dans le Model. Elles sont volontairement simples : elles couvrent le cas
# d'usage principal de chaque type (pas tous les cas particuliers).

def _cpp():
    """Lazy import pour éviter une boucle circulaire."""
    from .cpp import get_core
    return get_core()


@register_predict_fn('mlp')
def _predict_mlp(weights, config, X):
    """Inférence MLP : forward pass couches denses."""
    X = np.asarray(X, dtype=np.float64)
    # Reconstruction des listes depuis le dict aplati (format .gy)
    n_layers = config['n_layers']
    W_list = [weights[f'W_{i}'] for i in range(n_layers)]
    b_list = [weights[f'b_{i}'] for i in range(n_layers)]
    multiclass = config.get('multiclass', False)
    a = X
    for i in range(n_layers):
        z = a @ W_list[i] + b_list[i]
        if i < n_layers - 1:
            a = np.maximum(0.0, z)  # relu
        else:
            if multiclass:
                e = np.exp(z - z.max(axis=1, keepdims=True))
                a = e / e.sum(axis=1, keepdims=True)  # softmax
            else:
                a = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))  # sigmoid
    return a


@register_predict_fn('rnn')
def _predict_rnn(weights, config, X):
    """Inférence RNN vanilla (mono ou multi-couche) : forward sur (batch, seq_len, input_size)."""
    C = _cpp()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[np.newaxis, ...]
    hidden_size = config['hidden_size']
    seq_len = config.get('seq_len', X.shape[1])
    num_layers = config.get('num_layers', 1)
    W_hy = weights['W_hy']; b_y = weights['b_y']
    preds = []
    for i in range(X.shape[0]):
        h_list = [np.zeros((1, hidden_size)) for _ in range(num_layers)]
        for t in range(seq_len):
            x_t = X[i, t].reshape(1, -1)
            layer_input = x_t
            for l in range(num_layers):
                if num_layers == 1:
                    W_xh = weights['W_xh']
                    W_hh = weights['W_hh']
                    b_h = weights['b_h']
                else:
                    W_xh = weights[f'W_xh_{l}']
                    W_hh = weights[f'W_hh_{l}']
                    b_h = weights[f'b_h_{l}']
                h_list[l] = C.tanh(C.matmul(layer_input, W_xh) + C.matmul(h_list[l], W_hh) + b_h)
                layer_input = h_list[l]
        y_pred = C.sigmoid(C.matmul(h_list[-1], W_hy) + b_y)
        preds.append(y_pred[0, 0])
    return np.array(preds).reshape(-1, 1)


@register_predict_fn('cnn')
def _predict_cnn(weights, config, X):
    """Inférence CNN 2D/N-D (mono ou multi-couches) : conv + relu + FC + relu."""
    C = _cpp()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[np.newaxis, ...]
    w_fc = weights['w_fc']; b_fc = weights['b_fc']
    conv_fn = C.convolve2d if config.get('dimensions', 2) == 2 else C.convolve_nd

    # Handle both list format (kernels=[arr, ...]) and indexed format (kernels_0, kernels_1, ...)
    if 'kernels' in weights and isinstance(weights['kernels'], list):
        kernels = weights['kernels']
        biases = weights['biases']
    else:
        num_conv = config.get('num_conv_layers', 1)
        kernels = [weights[f'kernels_{i}'] for i in range(num_conv)]
        biases = [weights[f'biases_{i}'] for i in range(num_conv)]

    preds = []
    for i in range(X.shape[0]):
        img = X[i]
        x = img
        for li in range(len(kernels)):
            conv = conv_fn(x, kernels[li]) + biases[li]
            x = C.relu(conv)
        flat = x.flatten().reshape(1, -1)
        out = C.relu(C.matmul(flat, w_fc) + b_fc)
        preds.append(out[0, 0])
    return np.array(preds).reshape(-1, 1)


@register_predict_fn('transformer')
def _predict_transformer(weights, config, X):
    """Inférence Transformer : self-attention complet + FFN + pooling.

    Le moteur C++ ne supportant que le matmul 2D, chaque échantillon du
    batch est traité individuellement pour garantir la compatibilité.
    """
    C = _cpp()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[np.newaxis, ...]

    embed = weights.get('embedding')
    if embed is None:
        return np.zeros((X.shape[0], 1))

    seq_len = X.shape[1]
    num_blocks = config.get('num_blocks', 1)
    batch_size = X.shape[0]

    preds = []
    for sample_idx in range(batch_size):
        # (seq_len, embed_dim) — toujours 2D pour le moteur C++
        x = embed[X[sample_idx].astype(int)]

        for b_idx in range(num_blocks):
            # Support both flat keys (num_blocks=1) and indexed keys (num_blocks>1)
            if num_blocks == 1:
                wq = weights.get('W_q')
                wk = weights.get('W_k')
                wv = weights.get('W_v')
                gamma_attn = weights.get('gamma_attn')
                beta_attn = weights.get('beta_attn')
                wff1 = weights.get('W_ff1')
                bff1 = weights.get('b_ff1')
                wff2 = weights.get('W_ff2')
                bff2 = weights.get('b_ff2')
                gamma_ff = weights.get('gamma_ff')
                beta_ff = weights.get('beta_ff')
            else:
                wq = weights.get(f'W_q_{b_idx}')
                wk = weights.get(f'W_k_{b_idx}')
                wv = weights.get(f'W_v_{b_idx}')
                gamma_attn = weights.get(f'gamma_attn_{b_idx}')
                beta_attn = weights.get(f'beta_attn_{b_idx}')
                wff1 = weights.get(f'W_ff1_{b_idx}')
                bff1 = weights.get(f'b_ff1_{b_idx}')
                wff2 = weights.get(f'W_ff2_{b_idx}')
                bff2 = weights.get(f'b_ff2_{b_idx}')
                gamma_ff = weights.get(f'gamma_ff_{b_idx}')
                beta_ff = weights.get(f'beta_ff_{b_idx}')

            if wq is None or wk is None or wv is None:
                break

            d = x.shape[-1]
            # Self-attention (2D)
            Q = C.matmul(x, wq)
            K = C.matmul(x, wk)
            V = C.matmul(x, wv)
            attn_scores = C.matmul(Q, K.T) / math.sqrt(d)
            attn_weights = C.softmax(attn_scores)
            attn_output = C.matmul(attn_weights, V)

            # Add & Norm (attention)
            x_attn = x + attn_output
            x_norm = C.layer_norm(x_attn)
            if gamma_attn is not None and beta_attn is not None:
                x = x_norm * gamma_attn + beta_attn
            else:
                x = x_norm

            # FFN
            if wff1 is not None and wff2 is not None:
                b1 = np.asarray(bff1).flatten().reshape(1, -1) if bff1 is not None else 0
                b2 = np.asarray(bff2).flatten().reshape(1, -1) if bff2 is not None else 0
                ff1 = C.relu(C.matmul(x, wff1) + b1)
                ff2 = C.matmul(ff1, wff2) + b2

                # Add & Norm (FFN)
                x_ff = x + ff2
                x_ff_norm = C.layer_norm(x_ff)
                if gamma_ff is not None and beta_ff is not None:
                    x = x_ff_norm * gamma_ff + beta_ff
                else:
                    x = x_ff_norm

        # Mean pooling -> classify
        pooled = C.mean_axis(x, 0).reshape(1, -1)
        W_cls = weights.get('W_cls')
        b_cls = weights.get('b_cls')
        if W_cls is None or b_cls is None:
            preds.append(0)
            continue
        b_cls_r = np.asarray(b_cls).flatten().reshape(1, -1)
        logits = C.matmul(pooled, W_cls) + b_cls_r
        probs = C.softmax(logits)
        preds.append(int(np.argmax(probs[0])))

    return np.array(preds)


@register_predict_fn('gan')
def _predict_gan(weights, config, X):
    """Inférence GAN : génère des échantillons depuis le générateur.
    X est interprété comme du bruit latent (ou ignoré si None).
    Format indexé uniquement (G_W_0/G_W_1/…).
    """
    C = _cpp()
    latent_dim = weights['G_W_0'].shape[0]
    if X is not None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2 and X.shape[1] == latent_dim:
            z = X
        else:
            z = np.random.randn(1, latent_dim)
    else:
        z = np.random.randn(1, latent_dim)
    # Forward du générateur
    def _bias_row(b, cols):
        b = np.asarray(b).flatten()
        return b.reshape(1, cols)
    li = 0
    while f'G_W_{li}' in weights:
        li += 1
    n_layers = li
    h = z
    for layer_i in range(n_layers):
        W = weights[f'G_W_{layer_i}']
        b = _bias_row(weights[f'G_b_{layer_i}'], W.shape[1])
        z_l = C.matmul(h, W) + b
        if layer_i < n_layers - 1:
            h = C.leaky_relu(z_l, 0.01)
        else:
            out = z_l
    if config.get('apply_tanh', True):
        out = np.tanh(out)
    return out


@register_predict_fn('ldm')
def _predict_ldm(weights, config, X):
    """Inférence LDM : prédit le bruit depuis (x_noisy, class_embedding).
    X est l'ID de classe (ou ignoré).
    Utilise le class_embedding appris et les couches indexées W_0/b_0/…
    """
    C = _cpp()
    num_classes = config.get('num_classes', 5)
    input_dim = config.get('input_dim', 8)
    class_id = int(X) if X is not None else 0
    # Embedding de classe
    if 'class_embedding' in weights:
        class_emb = np.asarray(weights['class_embedding'])
        c_emb = class_emb[class_id % num_classes].reshape(1, -1)
    else:
        emb_dim = config.get('class_embedding_dim', num_classes)
        c_emb = np.zeros((1, emb_dim))
        c_emb[0, class_id % num_classes] = 1.0
    # Bruit initial
    x = np.random.randn(1, input_dim)
    x_concat = np.concatenate([x, c_emb], axis=1)
    # Couches indexées W_0, b_0, W_1, b_1, …
    li = 0
    while f'W_{li}' in weights:
        li += 1
    n_layers = li
    h = x_concat
    for layer_i in range(n_layers):
        W = weights[f'W_{layer_i}']
        b = np.asarray(weights[f'b_{layer_i}']).flatten().reshape(1, W.shape[1])
        z = C.matmul(h, W) + b
        if layer_i < n_layers - 1:
            h = C.relu(z)
        else:
            h = z
    pred_noise = h
    return pred_noise


@register_predict_fn('slm')
def _predict_slm(weights, config, X):
    """Inférence SLM : prédit la classe d'une séquence.
    X est une séquence d'indices (1D ou 2D).
    """
    C = _cpp()
    X = np.asarray(X).astype(int).flatten()
    embed = weights.get('embedding')
    if embed is None:
        return np.array([0])
    num_blocks = config.get('num_blocks', 1)
    seq = embed[X]
    x = seq
    for b_idx in range(num_blocks):
        W_q = weights.get(f'W_q_{b_idx}') if num_blocks > 1 else weights.get('W_q')
        W_k = weights.get(f'W_k_{b_idx}') if num_blocks > 1 else weights.get('W_k')
        W_v = weights.get(f'W_v_{b_idx}') if num_blocks > 1 else weights.get('W_v')
        W_ff1 = weights.get(f'W_ff1_{b_idx}') if num_blocks > 1 else weights.get('W_ff1')
        b_ff1 = weights.get(f'b_ff1_{b_idx}') if num_blocks > 1 else weights.get('b_ff1')
        W_ff2 = weights.get(f'W_ff2_{b_idx}') if num_blocks > 1 else weights.get('W_ff2')
        b_ff2 = weights.get(f'b_ff2_{b_idx}') if num_blocks > 1 else weights.get('b_ff2')
        gamma_attn = weights.get(f'gamma_attn_{b_idx}') if num_blocks > 1 else weights.get('gamma_attn')
        beta_attn = weights.get(f'beta_attn_{b_idx}') if num_blocks > 1 else weights.get('beta_attn')
        gamma_ff = weights.get(f'gamma_ff_{b_idx}') if num_blocks > 1 else weights.get('gamma_ff')
        beta_ff = weights.get(f'beta_ff_{b_idx}') if num_blocks > 1 else weights.get('beta_ff')
        if W_q is not None and W_k is not None and W_v is not None:
            d = x.shape[-1]
            q = C.matmul(x, W_q); k = C.matmul(x, W_k); v = C.matmul(x, W_v)
            scores = C.matmul(q, k.T) / math.sqrt(d)
            attn = C.softmax(scores)
            ctx = C.matmul(attn, v)
            x = x + ctx
            x_norm = C.layer_norm(x)
            if gamma_attn is not None and beta_attn is not None:
                x = x_norm * gamma_attn + beta_attn
            else:
                x = x_norm
        if W_ff1 is not None and W_ff2 is not None:
            b1 = np.asarray(b_ff1).flatten().reshape(1, -1) if b_ff1 is not None else 0
            b2 = np.asarray(b_ff2).flatten().reshape(1, -1) if b_ff2 is not None else 0
            h = C.relu(C.matmul(x, W_ff1) + b1)
            ff_out = C.matmul(h, W_ff2) + b2
            x = x + ff_out
            x_ff_norm = C.layer_norm(x)
            if gamma_ff is not None and beta_ff is not None:
                x = x_ff_norm * gamma_ff + beta_ff
            else:
                x = x_ff_norm
    pooled = x.mean(axis=0, keepdims=True)
    W_cls = weights.get('W_cls')
    b_cls = weights.get('b_cls')
    if W_cls is None or b_cls is None:
        return np.array([0])
    b_cls_r = np.asarray(b_cls).flatten().reshape(1, -1)
    logits = C.matmul(pooled, W_cls) + b_cls_r
    probs = C.softmax(logits)
    return np.array([int(np.argmax(probs[0]))])


@register_predict_fn('image_classifier')
def _predict_image_classifier(weights, config, X):
    """Inférence classifieur d'images : FC variable + softmax.
    X est une image (H, W) ou un batch (N, H, W) ou déjà flatten (N, F).
    Format multi-couches uniquement (W_0/W_1/…/W_out).
    """
    C = _cpp()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[np.newaxis, ...]
    hidden_layers = config.get('hidden_layers')
    if hidden_layers is None:
        hidden_layers = [config.get('hidden_dim', 64)]  # default single layer
    n_hidden = len(hidden_layers)
    input_dim = config.get('feature_dim', weights.get('feature_dim', 128))
    preds = []
    for i in range(X.shape[0]):
        feat = X[i].flatten().reshape(1, -1)
        # Padding/troncature à la dimension d'entrée
        if feat.shape[1] < input_dim:
            feat = np.pad(feat, ((0, 0), (0, input_dim - feat.shape[1])))
        elif feat.shape[1] > input_dim:
            feat = feat[:, :input_dim]
        a = feat
        for j in range(n_hidden):
            z = C.matmul(a, weights[f'W_{j}']) + weights[f'b_{j}']
            a = C.relu(z)
        out = C.matmul(a, weights['W_out']) + weights['b_out']
        probs = C.softmax(out)
        preds.append(int(np.argmax(probs[0])))
    return np.array(preds)


@register_predict_fn('speech_classifier')
def _predict_speech_classifier(weights, config, X):
    """Inférence classifieur de parole : conv1d -> relu -> pool -> [FC -> relu]* -> softmax.
    Supporte : avec ou sans convolution, avec ou sans couches FC cachées."""
    C = _cpp()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[np.newaxis, ...]

    conv_W = weights.get('conv_W')
    conv_b = weights.get('conv_b')
    kernel_size = int(config.get('conv_kernel_size', 3))
    dilation = int(config.get('conv_dilation', 1))

    hidden_layers = config.get('hidden_layers')
    n_fc = len(hidden_layers) if hidden_layers else 0

    # Déterminer la dimension d'entrée des couches FC
    if conv_W is not None:
        fc_input_dim = int(conv_W.shape[0])  # conv_out_channels
    elif n_fc > 0:
        first_fc = weights.get('W_fc_0')
        fc_input_dim = int(first_fc.shape[0]) if first_fc is not None else int(config.get('hidden_dim', 64))
    else:
        out_w = weights.get('W_fc_out')
        fc_input_dim = int(out_w.shape[0]) if out_w is not None else int(config.get('hidden_dim', 64))

    preds = []
    for i in range(X.shape[0]):
        x_2d = X[i].reshape(1, -1)
        if conv_W is not None:
            out_channels = int(conv_W.shape[0])
            out_len = max(1, x_2d.shape[1] - (kernel_size - 1) * dilation)
            conv_out = np.zeros((out_channels, out_len))
            for co in range(out_channels):
                for j in range(out_len):
                    total = 0.0
                    for ki in range(kernel_size):
                        idx = j + ki * dilation
                        if idx < x_2d.shape[1]:
                            total += x_2d[0, idx] * conv_W[co, 0, ki]
                    conv_out[co, j] = total + (float(conv_b[co]) if conv_b is not None else 0.0)
            conv_act = np.maximum(0.0, conv_out)
            pooled = conv_act.mean(axis=1).reshape(1, -1)
        else:
            # Padding/troncature à la dimension attendue
            if x_2d.shape[1] < fc_input_dim:
                pooled = np.pad(x_2d, ((0, 0), (0, fc_input_dim - x_2d.shape[1])))
            else:
                pooled = x_2d[:, :fc_input_dim]

        a = pooled
        for j in range(n_fc):
            w_j = weights.get(f'W_fc_{j}')
            b_j = weights.get(f'b_fc_{j}')
            if w_j is None or b_j is None:
                break
            z = np.asarray(C.matmul(a, w_j) + b_j)
            a = np.maximum(0.0, z)

        fc_out_w = weights.get('W_fc_out') if weights.get('W_fc_out') is not None else weights.get('W_fc')
        fc_out_b = weights.get('b_fc_out') if weights.get('b_fc_out') is not None else weights.get('b_fc')
        if fc_out_w is None or fc_out_b is None:
            preds.append(0)
            continue
        z_fc = np.asarray(C.matmul(a, fc_out_w) + fc_out_b)
        out = C.softmax(z_fc)
        preds.append(int(np.argmax(out[0])))
    return np.array(preds)


# ============================================================================
# Fonctions d'extraction des poids et config depuis les résultats train_xxx
# ============================================================================

def _extract_mlp(result):
    m = result['model']
    # Aplatir les listes de poids/biais en dict individuel pour le format .gy
    w = {}
    for i, (W, b) in enumerate(zip(m['weights'], m['biases'])):
        w[f'W_{i}'] = np.asarray(W, dtype=np.float64)
        w[f'b_{i}'] = np.asarray(b, dtype=np.float64)
    config = {
        'hidden_sizes': m['hidden_sizes'],
        'n_features': m['n_features'],
        'n_out': m['n_out'],
        'n_classes': m['n_classes'],
        'multiclass': m['multiclass'],
        'unique_classes': m['unique_classes'],
        'n_layers': len(m['weights']),
    }
    return (w, config)

def _extract_rnn(result):
    m = result['model']
    num_layers = m.get('num_layers', 1)
    w = {}
    if num_layers == 1:
        w = {k: m[k] for k in ['W_xh', 'W_hh', 'b_h', 'W_hy', 'b_y']}
    else:
        for l in range(num_layers):
            w[f'W_xh_{l}'] = m[f'W_xh_{l}']
            w[f'W_hh_{l}'] = m[f'W_hh_{l}']
            w[f'b_h_{l}'] = m[f'b_h_{l}']
        w['W_hy'] = m['W_hy']
        w['b_y'] = m['b_y']
    return (w,
            {'hidden_size': m['hidden_size'], 'seq_len': m['seq_len'],
             'input_size': m['input_size'], 'num_layers': num_layers})

def _extract_cnn(result):
    m = result['model']
    # Normalize keys: singular -> plural for backward compat
    if 'kernel' in m and 'kernels' not in m:
        m['kernels'] = m.pop('kernel')
    if 'bias' in m and 'biases' not in m:
        m['biases'] = m.pop('bias')
    w = {'kernels': m['kernels'], 'biases': m['biases'],
         'w_fc': m['w_fc'], 'b_fc': m['b_fc']}
    config = {
        'input_shape': list(m.get('input_shape', m.get('volume_shape', (5, 5)))),
        'kernel_shape': list(m.get('kernel_shape', (3, 3))),
        'dimensions': m.get('dimensions', 2),
        'num_conv_layers': m.get('num_conv_layers', 1),
    }
    # Préserver conv_shape et accuracy pour compatibilité load_cnn2d/load_cnn_nd
    if 'conv_shape' in m:
        config['conv_shape'] = list(m['conv_shape'])
    if 'accuracy' in m:
        config['accuracy'] = m['accuracy']
    if 'volume_shape' in m:
        config['volume_shape'] = list(m['volume_shape'])
    return (w, config)

def _extract_transformer(result):
    m = result['model']
    weights = {}
    num_blocks = m.get('num_blocks', 1)
    if num_blocks == 1:
        for k in ['embedding', 'W_q', 'W_k', 'W_v', 'W_ff1', 'W_ff2',
                  'b_ff1', 'b_ff2', 'W_cls', 'b_cls']:
            if k in m:
                weights[k] = m[k]
    else:
        weights['embedding'] = m['embedding']
        weights['W_q_list'] = m['W_q_list']
        weights['W_k_list'] = m['W_k_list']
        weights['W_v_list'] = m['W_v_list']
        weights['W_ff1_list'] = m['W_ff1_list']
        weights['b_ff1_list'] = m['b_ff1_list']
        weights['W_ff2_list'] = m['W_ff2_list']
        weights['b_ff2_list'] = m['b_ff2_list']
        weights['W_cls'] = m['W_cls']
        weights['b_cls'] = m['b_cls']
        weights['num_blocks'] = num_blocks
    return (weights, {'seq_len': m.get('seq_len', 4),
                      'vocab_size': m.get('vocab_size', 6),
                      'embed_dim': m.get('embed_dim', 8),
                      'num_blocks': num_blocks})

def _extract_gan(result):
    m = result['model']
    weights = {}
    for k, v in m.items():
        if isinstance(v, np.ndarray):
            weights[k] = v
    config = {'latent_dim': m.get('latent_dim', 4),
              'data_dim': m.get('data_dim', 1),
              'hidden_dim': m.get('hidden_dim', 16)}
    # Préserver les métadonnées pour load_gan_*
    for key in ('volume_shape', 'image_shape', 'generator_layers',
                'discriminator_layers'):
        if key in m:
            config[key] = m[key]
    return weights, config

def _extract_ldm(result):
    m = result['model']
    weights = {k: v for k, v in m.items() if isinstance(v, np.ndarray)}
    class_emb = m.get('class_embedding')
    class_emb_dim = class_emb.shape[1] if class_emb is not None and hasattr(class_emb, 'shape') else None
    config = {'timesteps': m.get('timesteps', 50),
              'num_classes': m.get('num_classes', 5),
              'input_dim': m.get('input_dim', 8),
              'class_embedding_dim': class_emb_dim}
    # Préserver hidden_sizes (liste, pas ndarray) dans la config
    hs = m.get('hidden_sizes')
    if hs is not None:
        config['hidden_sizes'] = hs
    return (weights, config)

def _extract_slm(result):
    m = result['model']
    # Les poids sont dans 'network_params' (sous-dict de ndarrays)
    params = m.get('network_params', {})
    weights = {k: v for k, v in params.items() if isinstance(v, np.ndarray)}
    # Inclure num_blocks dans les poids pour le predict_fn
    num_blocks = params.get('num_blocks', 1)
    if num_blocks > 1:
        weights['num_blocks'] = num_blocks
    config = {'vocab_size': m.get('vocab_size', 16),
              'seq_len': m.get('seq_len', 4),
              'embed_dim': m.get('embed_dim', 16),
              'num_blocks': num_blocks}
    # Préserver les métadonnées pour load_slm
    if 'class_names' in m:
        config['class_names'] = m['class_names']
    if 'preprocessor_params' in m:
        config['preprocessor_params'] = m['preprocessor_params']
    if 'num_classes' in m:
        config['num_classes'] = m['num_classes']
    if 'ff_dim' in m:
        config['ff_dim'] = m['ff_dim']
    return weights, config

def _extract_image_classifier(result):
    m = result['model']
    hidden_layers = m.get('hidden_layers')
    hidden_dim = m.get('hidden_dim', 32)

    if hidden_layers is None:
        # Legacy format: W1/b1 (hidden) + W2/b2 (output) → indexed format
        hidden_layers = [hidden_dim]
        w = {
            'W_0': m['W1'], 'b_0': m['b1'],
            'W_out': m['W2'], 'b_out': m['b2'],
        }
    else:
        w = {k: v for k, v in m.items() if isinstance(v, np.ndarray)}

    config = {
        'num_classes': m.get('num_classes', 3),
        'feature_dim': m.get('feature_dim', 128),
        'hidden_dim': hidden_dim,
        'hidden_layers': hidden_layers,
    }
    # Préserver class_names et accuracy pour load_image_classifier
    if 'class_names' in m:
        config['class_names'] = m['class_names']
    if 'accuracy' in m:
        config['accuracy'] = m['accuracy']
    return (w, config)


def _extract_speech_classifier(result):
    m = result['model']
    weights = {}

    # Extraire les poids conv depuis conv_params (sous-dict)
    conv_params = m.get('conv_params', {})
    if isinstance(conv_params, dict):
        for ck in ('W', 'b'):
            if ck in conv_params and isinstance(conv_params[ck], np.ndarray):
                weights[f'conv_{ck}'] = conv_params[ck]
        for meta_key in ('kernel_size', 'out_channels', 'in_channels', 'dilation'):
            if meta_key in conv_params:
                weights[f'conv_{meta_key}'] = conv_params[meta_key]
    elif isinstance(conv_params, (list, tuple)) and len(conv_params) > 0:
        # Ancien format : conv_params était une liste
        if isinstance(conv_params[0], np.ndarray):
            weights['conv_W'] = conv_params[0]
            weights['conv_b'] = conv_params[1] if len(conv_params) > 1 else np.zeros(1)

    # Extraire les poids FC
    hidden_layers = m.get('hidden_layers')
    if hidden_layers and len(hidden_layers) > 0:
        n_fc = len(hidden_layers)
        for i in range(n_fc):
            w = m.get(f'W_fc_{i}')
            b = m.get(f'b_fc_{i}')
            if w is not None:
                weights[f'W_fc_{i}'] = w
            if b is not None:
                weights[f'b_fc_{i}'] = b
        w_out = m.get('W_fc_out')
        b_out = m.get('b_fc_out')
        if w_out is not None:
            weights['W_fc_out'] = w_out
        if b_out is not None:
            weights['b_fc_out'] = b_out
    else:
        # Format sans hidden layers : W_fc/b_fc sont la sortie directe
        w_fc = m.get('W_fc')
        b_fc = m.get('b_fc')
        if w_fc is not None:
            weights['W_fc_out'] = w_fc
        if b_fc is not None:
            weights['b_fc_out'] = b_fc

    # Config
    config = {
        'feature_dim': m.get('feature_dim', 26),
        'hidden_dim': m.get('hidden_dim', 64),
        'num_classes': m.get('num_classes', 3),
        'class_names': m.get('class_names', ["voyelle", "consonne", "silence"]),
        'hidden_layers': hidden_layers if hidden_layers is not None else [],
        'accuracy': m.get('accuracy', 0),
        'num_features': m.get('num_features', 26),
    }
    for ck in ('conv_in_channels', 'conv_out_channels', 'conv_kernel_size', 'conv_dilation'):
        if ck in m:
            config[ck] = m[ck]
    # Propager les métadonnées conv depuis conv_params si pas déjà dans config
    for meta_key in ('kernel_size', 'out_channels', 'in_channels', 'dilation'):
        ck = f'conv_{meta_key}'
        if meta_key in conv_params and ck not in config:
            config[ck] = conv_params[meta_key]
    return (weights, config)


# ============================================================================
# Enregistrement automatique au chargement du module
# ============================================================================

def _register_all():
    """Enregistre tous les types supportés. Appelé paresseusement."""
    if _REGISTRY:
        return
    try:
        from .train import (train_rnn, train_cnn2d, train_transformer,
                            train_gan_nd, train_ldm_image,
                            train_slm_next_word, train_image_classifier,
                            train_mlp, train_speech_classifier)
    except ImportError as e:
        logger.warning("Impossible d'importer les modules train_xxx : %s", e)
        return

    register_model_type('mlp', train_mlp, _predict_mlp,
                        _extract_mlp, _extract_mlp, 'mini_mlp')
    register_model_type('rnn', train_rnn, _predict_rnn,
                        _extract_rnn, _extract_rnn, 'mini_rnn')
    register_model_type('cnn', train_cnn2d, _predict_cnn,
                        _extract_cnn, _extract_cnn, 'mini_cnn')
    register_model_type('transformer', train_transformer, _predict_transformer,
                        _extract_transformer, _extract_transformer, 'mini_transformer')
    register_model_type('gan', train_gan_nd, _predict_gan,
                        _extract_gan, _extract_gan, 'mini_gan')
    register_model_type('ldm', train_ldm_image, _predict_ldm,
                        _extract_ldm, _extract_ldm, 'mini_ldm')
    register_model_type('slm', train_slm_next_word, _predict_slm,
                        _extract_slm, _extract_slm, 'mini_slm')
    register_model_type('image_classifier', train_image_classifier,
                        _predict_image_classifier,
                        _extract_image_classifier, _extract_image_classifier,
                        'mini_image_classifier')
    register_model_type('speech_classifier', train_speech_classifier,
                        _predict_speech_classifier,
                        _extract_speech_classifier, _extract_speech_classifier,
                        'mini_speech_classifier')


# ============================================================================
# Classe Trainer
# ============================================================================

class Trainer:
    """Point d'entrée unifié pour entraîner tous les modèles IA.

    Args:
        verbose: si True, affiche la progression (callback ProgressPrinter).
        callbacks: liste de callbacks additionnels à ajouter.
        default_save_dir: répertoire de sauvegarde par défaut.

    Exemple :
        >>> trainer = Trainer(verbose=True)
        >>> model = trainer.train(type='rnn', epochs=500, lr=0.01)
        >>> y = model.predict(X_test)
        >>> model.save('rnn.gy')

        >>> # Rechargement
        >>> model = Trainer.load('rnn.gy')
        >>> y = model.predict(X_test)
    """

    SUPPORTED_TYPES = ['mlp', 'rnn', 'cnn', 'transformer', 'gan', 'ldm',
                        'slm', 'image_classifier', 'speech_classifier']

    def __init__(self, verbose: bool = True,
                 callbacks: Optional[List[Callback]] = None,
                 default_save_dir: Optional[str] = None):
        _register_all()
        self.verbose = verbose
        self.user_callbacks = callbacks or []
        self.default_save_dir = default_save_dir or MODELS_DIR
        # Etat interne mis à jour pendant train()
        self.model_type: Optional[str] = None
        self.epochs: int = 0
        self.lr: float = 0.0
        self.stop_training: bool = False
        self.history: List[Dict[str, float]] = []
        self.model_path: Optional[str] = None
        self._current_model_data: Optional[Dict[str, Any]] = None
        self._current_config: Optional[Dict[str, Any]] = None
        self._current_extract_weights: Optional[Callable] = None

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------
    def train(self, type: str, X=None, y=None, dataset=None,
              dataset_target: Optional[str] = None,
              epochs: int = 1000,
              lr: float = 0.01, batch_size: Optional[int] = None,
              seed: Optional[int] = 42,
              callbacks: Optional[List[Callback]] = None,
              save: bool = True, save_path: Optional[str] = None,
              early_stopping_patience: int = 0,
              **kwargs) -> Model:
        """Entraîne un modèle du type spécifié.

        Args:
            type: type de modèle ('mlp', 'rnn', 'cnn', 'transformer', 'gan',
                  'ldm', 'slm', 'image_classifier', 'speech_classifier').
            X, y: données d'entraînement (None = données démo).
            dataset: chemin vers un fichier ou dossier de données.
                     Formats fichier : .csv, .tsv, .json, .jsonl, .npy,
                     .npz, .txt, .h5, .parquet, .xlsx, .pkl, .gz
                     Dossier : structure label/image ou images brutes.
                     Si fourni, X et y sont ignorés.
            dataset_target: nom de la colonne cible (pour fichiers tabulaires).
            epochs: nombre d'époques.
            lr: taux d'apprentissage.
            batch_size: taille de batch (ignoré par certains modèles).
            seed: graine aléatoire.
            callbacks: liste de callbacks additionnels (Keras-like).
            save: si True, sauvegarde le modèle en .gy à la fin.
            save_path: chemin de sauvegarde. Si None, généré automatiquement.
            early_stopping_patience: patience EarlyStopping (0 = désactivé).
            **kwargs: hyperparams additionnels passés au train_xxx.

            Paramètres de profondeur (passés via **kwargs) :
              mlp  : hidden_sizes=[64, 32]       (liste libre, par défaut [64, 32])
              rnn  : num_layers=1                (couches récurrentes empilées)
              cnn  : num_conv_layers=1           (couches de convolution empilées)
              transformer : num_blocks=1         (blocs attention+FFN empilés)
              gan  : generator_layers=None, discriminator_layers=None
                     (listes d'ints, ex: [latent, 128, 64, data_dim])
              ldm  : hidden_sizes=None           (listes, ex: [128, 64])
              slm  : num_blocks=1                (blocs attention+FFN empilés)
              image_classifier : hidden_layers=None (liste, ex: [128, 64])

        Returns:
            Un objet Model avec predict_fn enregistrée.

        Examples:
            trainer.train(type='cnn', epochs=100)                       # démo interne
            trainer.train(type='cnn', dataset='mnist.csv', epochs=100)  # fichier CSV
            trainer.train(type='cnn', dataset='images/', epochs=100)    # dossier images
        """
        # --- Résolution du dataset -------------------------------------------
        if dataset is not None:
            from .dataset import load_dataset, load_folder
            if os.path.isdir(dataset):
                logger.info("Chargement dataset dossier : %s", dataset)
                X, y = load_folder(dataset)
            else:
                logger.info("Chargement dataset fichier : %s", dataset)
                result = load_dataset(dataset, target=dataset_target)
                if isinstance(result, tuple):
                    X, y = result
                else:
                    X, y = result, None
            logger.info("Dataset résolu : X=%s y=%s", X.shape,
                        y.shape if y is not None else None)
        # ---------------------------------------------------------------------

        if type not in _REGISTRY:
            raise ConfigurationError(
                f"Type de modèle inconnu : {type!r}. "
                f"Types supportés : {list(_REGISTRY.keys())}"
            )

        entry = _REGISTRY[type]
        train_fn = entry['train_fn']

        # Préparation de l'état
        self.model_type = type
        self.epochs = epochs
        self.lr = lr
        self.stop_training = False
        self.history = []
        self._current_extract_weights = entry['extract_weights']

        # Callbacks
        all_callbacks = list(self.user_callbacks)
        if callbacks:
            all_callbacks.extend(callbacks)
        if self.verbose and not any(isinstance(c, type(ProgressPrinter)) for c in all_callbacks):
            from .callbacks import ProgressPrinter
            # On n'ajoute ProgressPrinter que si l'utilisateur n'en a pas fourni un
            if not any(c.__class__.__name__ == 'ProgressPrinter' for c in all_callbacks):
                all_callbacks.append(ProgressPrinter(interval=max(1, epochs // 10)))
        if early_stopping_patience > 0:
            from .callbacks import EarlyStopping
            all_callbacks.append(EarlyStopping(patience=early_stopping_patience))

        cb_list = CallbackList(all_callbacks)
        cb_list.on_train_begin(self)

        # Construction des kwargs pour train_xxx
        train_kwargs = {
            'lr': lr,
            'epochs': epochs,
            'seed': seed,
            **entry['default_kwargs'],
            **kwargs,
        }
        if X is not None:
            train_kwargs['X'] = X
        if y is not None:
            train_kwargs['y'] = y
        # Résoudre le save_path pour le train function
        if save:
            if save_path is None:
                resolved_path = self._default_save_path(type)
            else:
                dirname = os.path.dirname(save_path)
                if not dirname:
                    resolved_path = os.path.join(MODELS_DIR, save_path)
                else:
                    resolved_path = save_path
                if not resolved_path.endswith('.gy'):
                    resolved_path += '.gy'
            train_kwargs['save_path'] = resolved_path
        else:
            resolved_path = None

        # Entraînement
        try:
            result = train_fn(**train_kwargs)
        except Exception as e:
            logger.error("Échec entraînement %s : %s", type, e)
            raise TrainingError(
                f"Échec entraînement {type} : {e}"
            )

        # Extraction des poids et config
        weights, config = entry['extract_weights'](result)

        # Construction du Model
        model = Model(type, config, weights, predict_fn=entry['predict_fn'])

        # Le train_xxx a sauvegardé avec son format, on re-sauvegarde avec
        # le format Model (type, config, weights_meta) sur le même fichier
        if resolved_path:
            model.save(resolved_path)
            logger.info("Modèle sauvegardé : %s", resolved_path)
            self.model_path = resolved_path
        else:
            self.model_path = None

        cb_list.on_train_end(self)
        return model

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> Model:
        """Charge un modèle .gy et retourne un Model prêt à inférer.

        Args:
            path: chemin du fichier .gy. Si le fichier n'existe pas
                  directement, cherche dans le dossier models/ du module.

        Returns:
            Model avec predict_fn récupérée depuis le registre.
        """
        _register_all()
        # Forcer l'extension .gy
        base, ext = os.path.splitext(path)
        if ext and ext != '.gy':
            path = base + '.gy'
        elif not ext:
            path = path + '.gy'
        # Résolution du chemin : chercher dans models/ si non trouvé
        if not os.path.isfile(path):
            candidate = os.path.join(MODELS_DIR, os.path.basename(path))
            if os.path.isfile(candidate):
                path = candidate
        return Model.load(path)

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------
    def _default_save_path(self, model_type: str) -> str:
        ensure_directories()
        entry = _REGISTRY.get(model_type, {})
        name = entry.get('default_save_name', f'mini_{model_type}')
        return os.path.join(self.default_save_dir, name)

    def _save_current(self, filepath: str):
        """Appelé par ModelCheckpoint pendant l'entraînement.
        Sauvegarde l'état courant du modèle en .gy.
        """
        if self._current_model_data is None or self._current_config is None:
            return
        if self._current_extract_weights is None:
            return
        # Conversion : _current_model_data est le dict 'model' retourné par train_xxx
        # On doit l'encapsuler comme un 'result' pour extract_weights
        result = {'model': self._current_model_data, 'config': self._current_config}
        weights, config = self._current_extract_weights(result)
        m = Model(self.model_type, config, weights)
        m.save(filepath)

    # ------------------------------------------------------------------
    # Représentation
    # ------------------------------------------------------------------
    def __repr__(self):
        return (f"Trainer(types={list(_REGISTRY.keys())}, "
                f"verbose={self.verbose})")
