"""
IA/train/rnn.py — Entraînement du RNN vanilla.

Architecture : RNN empilé (stacked) avec BPTT pour la classification de séquences binaires.
Supporte un nombre configurable de couches RNN via le paramètre ``num_layers``.
"""

import numpy as np
import pickle
import os
import logging

from ..config import get_config, MODELS_DIR, MODEL_EXTENSION, ensure_directories
from ..cpp import get_core

C = get_core()

logger = logging.getLogger(__name__)


def sigmoid(x):
    return C.sigmoid(x)

def sigmoid_deriv(x):
    return x * (1 - x)

def tanh(x):
    return C.tanh(x)

def tanh_deriv(x):
    return C.tanh_deriv(x)


def train_rnn(X=None, y=None, hidden_size=4, lr=0.1, epochs=1000,
              save_path=None, seed=42, grad_clip=1.0, num_layers=1):
    """
    Entraîne un RNN vanilla (empilé) pour la classification de séquences.

    Args:
        X: Données d'entrée (batch, seq_len, input_size).
           Par défaut : 2 séquences, 5 pas de temps, 2 features.
        y: Labels (batch, 1). Par défaut : [[1], [0]].
        hidden_size: Taille de l'état caché (identique pour toutes les couches).
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques maximum.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        grad_clip: Valeur maximale des gradients.
        num_layers: Nombre de couches RNN empilées (défaut 1).
            - Couche 0 : input_size  -> hidden_size
            - Couches 1..num_layers-1 : hidden_size -> hidden_size
            - Couche de sortie : last_hidden -> 1
            Quand num_layers=1, le comportement est identique à la version
            monocouche originale (clés du modèle : W_xh, W_hh, b_h).
            Pour num_layers>1 les paramètres sont sauvegardés sous la forme
            W_xh_0, W_xh_1, …, W_hh_0, W_hh_1, …, b_h_0, b_h_1, … .

    Returns:
        dict: Paramètres du modèle et métriques.
    """
    ensure_directories()

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"mini_rnn{MODEL_EXTENSION}")

    if X is None or y is None:
        X = np.array([
            [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]],
            [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
        ])
        y = np.array([[1], [0]])

    seq_len = X.shape[1]
    input_size = X.shape[2]

    # ------------------------------------------------------------------ #
    #  Initialisation des paramètres pour chaque couche RNN               #
    # ------------------------------------------------------------------ #
    W_xh_list = []
    W_hh_list = []
    b_h_list = []

    for l in range(num_layers):
        fan_in = input_size if l == 0 else hidden_size
        s = seed + 2 * l
        W_xh_list.append(C.scale(C.randn((fan_in, hidden_size), s), 0.1))
        W_hh_list.append(C.scale(C.randn((hidden_size, hidden_size), s + 1), 0.1))
        b_h_list.append(C.zeros((1, hidden_size)))

    W_hy = C.scale(C.randn((hidden_size, 1), seed + 2 * num_layers), 0.1)
    b_y = C.zeros((1, 1))

    history = []

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            # ---------------------------------------------------------- #
            #  Forward pass                                               #
            # ---------------------------------------------------------- #
            # hs_list[layer][t] = hidden state of layer at time-step t
            #   hs_list[layer][0] = initial zero state
            #   hs_list[layer][t+1] = state after processing input at t
            hs_list = []
            for l in range(num_layers):
                hs_list.append([C.zeros((1, hidden_size))])

            xs = []
            for t in range(seq_len):
                x_t = X[i, t].reshape(1, -1)
                xs.append(x_t)

                layer_input = x_t
                for l in range(num_layers):
                    h_new = tanh(
                        C.matmul(layer_input, W_xh_list[l])
                        + C.matmul(hs_list[l][-1], W_hh_list[l])
                        + b_h_list[l]
                    )
                    hs_list[l].append(h_new)
                    layer_input = h_new  # output of this layer -> input of next

            # Output from last hidden state of the top layer
            y_pred = sigmoid(C.matmul(hs_list[-1][-1], W_hy) + b_y)
            error = y[i] - y_pred
            loss = C.sum(C.pow(error, 2))
            total_loss += loss

            # ---------------------------------------------------------- #
            #  BPTT                                                       #
            # ---------------------------------------------------------- #
            dy = error * sigmoid_deriv(y_pred)
            dW_hy = C.matmul(hs_list[-1][-1].T, dy)
            db_y = dy

            dW_xh_list = [C.zeros(W_xh_list[l].shape) for l in range(num_layers)]
            dW_hh_list = [C.zeros(W_hh_list[l].shape) for l in range(num_layers)]
            db_h_list = [C.zeros(b_h_list[l].shape) for l in range(num_layers)]

            # Gradient flowing from the output into the top layer.
            # Applied at every time-step to match the original single-layer
            # BPTT pattern (backward compatible when num_layers == 1).
            dh_from_output = C.matmul(dy, W_hy.T)

            # Process layers from top to bottom.
            # dh_above_per_timestep[t] holds the gradient arriving at the
            # current layer from the layer above at time-step t.
            dh_above_per_timestep = [dh_from_output] * seq_len

            for l in reversed(range(num_layers)):
                dh_next = C.zeros((1, hidden_size))
                dh_below_ts = []  # gradient for layer below at each t

                for t in reversed(range(seq_len)):
                    h_t = hs_list[l][t + 1]
                    h_prev = hs_list[l][t]

                    if l == 0:
                        x_input_t = xs[t]
                    else:
                        x_input_t = hs_list[l - 1][t + 1]

                    dh = dh_above_per_timestep[t] + dh_next
                    dtanh = dh * tanh_deriv(h_t)
                    db_h_list[l] += dtanh
                    dW_xh_list[l] += C.matmul(x_input_t.T, dtanh)
                    dW_hh_list[l] += C.matmul(h_prev.T, dtanh)
                    dh_next = C.matmul(dtanh, W_hh_list[l].T)

                    if l > 0:
                        dh_below_ts.append(C.matmul(dtanh, W_xh_list[l].T))

                if l > 0:
                    dh_above_per_timestep = list(reversed(dh_below_ts))

            # Gradient clipping (W_xh, W_hh, W_hy only — same as original)
            for l in range(num_layers):
                dW_xh_list[l] = C.clip(dW_xh_list[l], -grad_clip, grad_clip)
                dW_hh_list[l] = C.clip(dW_hh_list[l], -grad_clip, grad_clip)
            dW_hy = C.clip(dW_hy, -grad_clip, grad_clip)

            # Parameter updates
            for l in range(num_layers):
                W_xh_list[l] += lr * dW_xh_list[l]
                W_hh_list[l] += lr * dW_hh_list[l]
                b_h_list[l] += lr * db_h_list[l]
            W_hy += lr * dW_hy
            b_y += lr * db_y

        avg_loss = total_loss / len(X)
        history.append(avg_loss)
        if epoch % 200 == 0:
            logger.info("Epoch %d, Loss: %.6f", epoch, avg_loss)

    # ------------------------------------------------------------------ #
    #  Évaluation                                                        #
    # ------------------------------------------------------------------ #
    correct = 0
    for i in range(len(X)):
        h_list = [C.zeros((1, hidden_size)) for _ in range(num_layers)]
        for t in range(seq_len):
            x_t = X[i, t].reshape(1, -1)
            layer_input = x_t
            for l in range(num_layers):
                h_list[l] = tanh(
                    C.matmul(layer_input, W_xh_list[l])
                    + C.matmul(h_list[l], W_hh_list[l])
                    + b_h_list[l]
                )
                layer_input = h_list[l]
        y_pred = sigmoid(C.matmul(h_list[-1], W_hy) + b_y)
        pred = 1 if y_pred[0, 0] > 0.5 else 0
        if pred == y[i][0]:
            correct += 1
    accuracy = correct / len(X) * 100

    # ------------------------------------------------------------------ #
    #  Sauvegarde du modèle                                              #
    # ------------------------------------------------------------------ #
    model = {
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'seq_len': seq_len,
        'input_size': input_size,
        'activations': {'hidden': 'tanh', 'output': 'sigmoid'},
        'accuracy': accuracy,
    }

    if num_layers == 1:
        # Backward-compatible keys for single-layer models
        model['W_xh'] = W_xh_list[0]
        model['W_hh'] = W_hh_list[0]
        model['b_h'] = b_h_list[0]
    else:
        for l in range(num_layers):
            model[f'W_xh_{l}'] = W_xh_list[l]
            model[f'W_hh_{l}'] = W_hh_list[l]
            model[f'b_h_{l}'] = b_h_list[l]

    model['W_hy'] = W_hy
    model['b_y'] = b_y

    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info("RNN sauvegardé dans %s (précision: %.1f%%)", save_path, accuracy)

    return {'model': model, 'save_path': save_path, 'accuracy': accuracy, 'history': history}