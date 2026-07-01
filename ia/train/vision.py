"""
IA/train/vision.py — Entraînement des modèles de classification perceptionnelle.

Modèles :
  - train_image_classifier : Classification d'images à partir de features extraites.
  - train_speech_classifier : Classification de phonèmes à partir de features audio.
"""

import numpy as np
import pickle
import os
import logging

from ..config import MODELS_DIR, MODEL_EXTENSION, ensure_directories
from ..cpp import get_core

C = get_core()

logger = logging.getLogger(__name__)


# ==================================================================
# Fonctions communes
# ==================================================================

def softmax(x):
    return C.softmax(x)


def relu(x):
    return C.relu(x)


def relu_deriv(x):
    return C.relu_deriv(x)


def xavier_init(shape, seed=42):
    return np.array(C.xavier_init(tuple(shape), seed))


# ==================================================================
# SimpleFeatureExtractor
# ==================================================================

class SimpleFeatureExtractor:
    """Extraction de features basiques à partir d'images RGB (numpy arrays)."""

    @staticmethod
    def extract_color_histogram(image, bins=16):
        """Histogramme RGB normalisé. image: (H, W, 3). Retourne 48 features."""
        features = []
        for c in range(min(image.shape[-1], 3)):
            channel = image[:, :, c].ravel()
            hist = C.histogram(channel, bins, 0, 256)
            hist = np.array(hist, dtype=float) / (sum(hist) + 1e-8)
            features.append(hist)
        return np.concatenate(features)

    @staticmethod
    def extract_gradient(image):
        """Magnitude de gradient moyenne (Sobel simplifié). Retourne 1 feature."""
        gray = np.mean(image.astype(float), axis=2)
        gx = C.diff_axis(gray, 1)
        gy = C.diff_axis(gray, 0)
        min_h = min(gx.shape[0], gy.shape[0])
        min_w = min(gx.shape[1], gy.shape[1])
        mag = C.sqrt(C.add(C.pow(gx[:min_h, :min_w], 2), C.pow(gy[:min_h, :min_w], 2)))
        return np.array([C.mean(mag)])

    @staticmethod
    def extract_lbp(image):
        """Texture LBP simplifiée (histogramme 16 bins). Retourne 16 features."""
        gray = np.mean(image.astype(float), axis=2)
        h, w = gray.shape
        lbp = np.zeros((h, w))
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray[i, j]
                code = 0
                neighbors = [
                    gray[i - 1, j - 1], gray[i - 1, j], gray[i - 1, j + 1],
                    gray[i, j + 1], gray[i + 1, j + 1], gray[i + 1, j],
                    gray[i + 1, j - 1], gray[i, j - 1],
                ]
                for k, n in enumerate(neighbors):
                    if n >= center:
                        code |= (1 << k)
                lbp[i, j] = code
        hist, _ = np.histogram(lbp, bins=16, range=(0, 256))
        hist = hist.astype(float) / (hist.sum() + 1e-8)
        return hist

    @staticmethod
    def extract_edges(image):
        """Détection de bords par différence simple sur grille 4x4. Retourne 16 features."""
        gray = np.mean(image.astype(float), axis=2)
        edges_h = np.abs(np.diff(gray, axis=0))
        edges_v = np.abs(np.diff(gray, axis=1))
        eh, ew = edges_h.shape
        evh, evw = edges_v.shape
        rh, rw = min(eh, evh), min(ew, evw)
        features = []
        for i in range(4):
            for j in range(4):
                hs = i * rh // 4
                he = (i + 1) * rh // 4
                ws = j * rw // 4
                we = (j + 1) * rw // 4
                region_h = edges_h[hs:he, :rw]
                region_v = edges_v[:rh, ws:we]
                features.append(np.mean(region_h) + np.mean(region_v))
        return np.array(features)

    @staticmethod
    def extract(image, target_dim=128):
        """Combine toutes les features et pad/tronque à target_dim."""
        color = SimpleFeatureExtractor.extract_color_histogram(image)
        grad = SimpleFeatureExtractor.extract_gradient(image)
        lbp = SimpleFeatureExtractor.extract_lbp(image)
        edges = SimpleFeatureExtractor.extract_edges(image)
        combined = np.concatenate([color, grad, lbp, edges])
        if len(combined) < target_dim:
            combined = np.pad(combined, (0, target_dim - len(combined)))
        else:
            combined = combined[:target_dim]
        return combined


# ==================================================================
# AudioFeatureExtractor
# ==================================================================

class AudioFeatureExtractor:
    """Extraction de features audio MFCC-like à partir de signaux 1D (numpy arrays)."""

    @staticmethod
    def extract_mfcc_features(audio_signal, num_mfcc=13):
        """MFCC simplifié. Retourne (n_frames, num_mfcc)."""
        frame_length = 256
        hop_length = 128
        n_fft = 256
        n_mels = 20
        signal = audio_signal.astype(float)
        sig_len = len(signal)

        if sig_len < frame_length:
            signal = np.pad(signal, (0, frame_length - sig_len))
            sig_len = frame_length

        n_frames = max(1, (sig_len - frame_length) // hop_length + 1)
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_length) / frame_length))

        # Filtre mel
        mel_low = 0.0
        mel_high = 2595.0 * np.log10(1.0 + 4000.0 / 700.0)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1)
        fft_bins = np.floor((n_fft + 1) * hz_points / 4000.0).astype(int)
        fft_bins = np.clip(fft_bins, 0, n_fft // 2)

        features = []
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            frame = signal[start:end] * window

            spectrum = C.fft_rfft(frame, n_fft)
            spectrum = spectrum[:n_fft // 2 + 1]

            # Énergies mel
            mel_energies = np.zeros(n_mels)
            for m in range(n_mels):
                left = fft_bins[m]
                center = fft_bins[m + 1]
                right = fft_bins[m + 2]
                for k in range(left, center):
                    if k < len(spectrum) and center > left:
                        mel_energies[m] += spectrum[k] * (k - left) / (center - left + 1e-8)
                for k in range(center, right):
                    if k < len(spectrum) and right > center:
                        mel_energies[m] += spectrum[k] * (right - k) / (right - center + 1e-8)

            log_mel = np.array(C.log(C.add_scalar(mel_energies, 1e-8)))

            # DCT-II simplifié
            mfcc = np.zeros(num_mfcc)
            for k in range(num_mfcc):
                for n in range(n_mels):
                    mfcc[k] += log_mel[n] * np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
            features.append(mfcc)

        return np.array(features)

    @staticmethod
    def extract_delta(features):
        """Delta de premier ordre. Retourne tableau de même forme."""
        if len(features) < 3:
            return np.zeros_like(features)
        delta = np.zeros_like(features)
        for t in range(1, len(features) - 1):
            delta[t] = 0.5 * (features[t + 1] - features[t - 1])
        delta[0] = delta[1]
        delta[-1] = delta[-2]
        return delta

    @staticmethod
    def extract(audio, sr=None):
        """Combine MFCC statiques + delta, moyenne temporelle. Retourne (26,)."""
        mfcc = AudioFeatureExtractor.extract_mfcc_features(audio, num_mfcc=13)
        delta = AudioFeatureExtractor.extract_delta(mfcc)
        combined = np.concatenate([mfcc, delta], axis=-1)
        return np.mean(combined, axis=0)


# ==================================================================
# TemporalConv1D
# ==================================================================

class TemporalConv1D:
    """Convolution 1D temporelle avec dilation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, seed=42):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.W = xavier_init((out_channels, in_channels, kernel_size), seed=seed)
        self.b = np.array(C.zeros((out_channels,)))
        self._cache = None

    def forward(self, x):
        """x: (in_channels, seq_len). Retourne (out_channels, out_seq_len)."""
        self._cache = {'x': x}
        return np.array(C.conv1d_forward(x, self.W, self.b, self.kernel_size, self.dilation))

    def backward(self, d_out):
        """d_out: (out_channels, out_seq_len). Retourne d_x, d_W, d_b."""
        x = self._cache['x']
        d_x, d_W, d_b = C.conv1d_backward(x, self.W, d_out, self.kernel_size, self.dilation)
        return np.array(d_x), np.array(d_W), np.array(d_b)

    def get_params(self):
        return {
            'in_channels': self.in_channels, 'out_channels': self.out_channels,
            'kernel_size': self.kernel_size, 'dilation': self.dilation,
            'W': self.W.copy(), 'b': self.b.copy(),
        }

    @classmethod
    def from_params(cls, params):
        layer = cls(params['in_channels'], params['out_channels'],
                    params['kernel_size'], params['dilation'], seed=0)
        layer.W = params['W']
        layer.b = params['b']
        return layer


# ==================================================================
# train_image_classifier
# ==================================================================

def train_image_classifier(X=None, y=None, num_classes=3, feature_dim=128,
                           hidden_dim=64, lr=0.01, epochs=500,
                           save_path=None, seed=42, hidden_layers=None):
    """
    Entraîne un classifieur d'images à partir de features extraites.

    Architecture configurable en profondeur :

      - ``hidden_layers=None`` (défaut) : input -> FC1 -> ReLU -> FC2 -> softmax.
        Comportement identique à l'original ; les paramètres sont sauvegardés
        sous les clés ``W1``, ``b1``, ``W2``, ``b2`` (rétrocompatible).

      - ``hidden_layers=[128, 64]`` : input -> FC(128) -> ReLU -> FC(64) -> ReLU
        -> FC(num_classes) -> softmax.  Les paramètres sont sauvegardés sous les
        clés ``W_0``, ``b_0``, ``W_1``, ``b_1``, …, ``W_out``, ``b_out``.

    Chaque couche cachée utilise l'activation ReLU ; la couche de sortie utilise
    softmax.  La descente de gradient stochastique (un échantillon à la fois)
    est appliquée à toutes les couches.

    Args:
        X: Features d'entrée (N, feature_dim).
        y: Labels (N,).
        num_classes: Nombre de classes.
        feature_dim: Dimension des features.
        hidden_dim: Dimension de l'unique couche cachée (utilisé uniquement
            si ``hidden_layers is None``).
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        hidden_layers: Liste d'entiers décrivant les dimensions des couches
            cachées successives.  Si ``None``, une seule couche cachée de
            dimension ``hidden_dim`` est utilisée (comportement par défaut).

    Returns:
        dict: Dictionnaire contenant ``'model'``, ``'save_path'``,
        ``'accuracy'`` et ``'history'``.
    """
    ensure_directories()

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"image_classifier{MODEL_EXTENSION}")

    if X is None or y is None:
        n_per_class = 50
        X = np.array(C.randn((num_classes * n_per_class, feature_dim), seed=seed)) * 0.1
        y = np.zeros(num_classes * n_per_class, dtype=int)
        for c in range(num_classes):
            start = c * n_per_class
            end = (c + 1) * n_per_class
            X[start:end, :feature_dim // num_classes] += c * 0.5
            y[start:end] = c

    # --- Détermination des dimensions des couches cachées ---
    if hidden_layers is None:
        layer_sizes = [hidden_dim]
    else:
        layer_sizes = list(hidden_layers)

    n_hidden = len(layer_sizes)

    # --- Initialisation des poids ---
    Ws = []
    bs = []
    prev_dim = feature_dim
    for i, size in enumerate(layer_sizes):
        Ws.append(xavier_init((prev_dim, size), seed=seed + i))
        bs.append(np.array(C.zeros((1, size))))
        prev_dim = size
    W_out = xavier_init((prev_dim, num_classes), seed=seed + n_hidden)
    b_out = np.array(C.zeros((1, num_classes)))

    best_loss = float('inf')
    best_params = None
    history = []

    for epoch in range(epochs):
        total_loss = 0
        indices = np.array(C.permutation(len(X), seed=seed + epoch))
        for idx in indices:
            xi = X[idx].reshape(1, -1)
            yi = int(y[idx])

            # ---- Forward ----
            z_list = []      # pre-activations des couches cachées
            a_list = [xi]    # activations (a_list[0]=input, a_list[i+1]=relu(z_i))
            a = xi
            for i in range(n_hidden):
                z = C.matmul(a, Ws[i]) + bs[i]
                z_list.append(z)
                a = relu(z)
                a_list.append(a)

            z_final = C.matmul(a, W_out) + b_out
            out = softmax(z_final)[0]

            loss = -np.log(float(out[yi]) + 1e-8)
            total_loss += loss

            # ---- Backward ----
            d_z_out = out.copy()
            d_z_out[yi] -= 1.0
            d_W_out = C.matmul(a_list[-1].T, d_z_out.reshape(1, -1))
            d_b_out = d_z_out.reshape(1, -1)
            d_a = C.matmul(d_z_out.reshape(1, -1), W_out.T)

            d_Ws = [None] * n_hidden
            d_bs = [None] * n_hidden
            for i in range(n_hidden - 1, -1, -1):
                d_z = d_a * relu_deriv(z_list[i])
                d_Ws[i] = C.matmul(a_list[i].T, d_z)
                d_bs[i] = d_z
                if i > 0:
                    d_a = C.matmul(d_z, Ws[i].T)

            # ---- Mise à jour SGD ----
            W_out += lr * d_W_out
            b_out += lr * d_b_out
            for i in range(n_hidden):
                Ws[i] = Ws[i] + lr * d_Ws[i]
                bs[i] = bs[i] + lr * d_bs[i]

            # ---- Clip ----
            W_out = C.clip(W_out, -1, 1)
            for i in range(n_hidden):
                Ws[i] = C.clip(Ws[i], -1, 1)

        avg_loss = total_loss / len(X)
        history.append(avg_loss)

        # Sauvegarde des meilleurs paramètres
        if avg_loss < best_loss:
            best_loss = avg_loss
            if hidden_layers is None:
                best_params = {
                    'W1': Ws[0].copy(), 'b1': bs[0].copy(),
                    'W2': W_out.copy(), 'b2': b_out.copy(),
                }
            else:
                best_params = {}
                for i in range(n_hidden):
                    best_params[f'W_{i}'] = Ws[i].copy()
                    best_params[f'b_{i}'] = bs[i].copy()
                best_params['W_out'] = W_out.copy()
                best_params['b_out'] = b_out.copy()

        if epoch % 100 == 0:
            logger.info("Epoch %d, Loss: %.6f", epoch, avg_loss)
        if avg_loss < 0.01:
            logger.info("Convergence a l'epoch %d", epoch)
            break

    # ---- Évaluation ----
    p = best_params
    correct = 0
    for i in range(len(X)):
        xi = X[i].reshape(1, -1)
        if hidden_layers is None:
            z1 = C.matmul(xi, p['W1']) + p['b1']
            a1 = relu(z1)
            z2 = C.matmul(a1, p['W2']) + p['b2']
            out = softmax(z2)[0]
        else:
            a = xi
            for j in range(n_hidden):
                z = C.matmul(a, p[f'W_{j}']) + p[f'b_{j}']
                a = relu(z)
            z = C.matmul(a, p['W_out']) + p['b_out']
            out = softmax(z)[0]
        pred = C.argmax(out)
        if pred == y[i]:
            correct += 1
    accuracy = correct / len(X) * 100

    # ---- Construction du dictionnaire modèle ----
    IMAGE_CLASSES = [f'Classe {i}' for i in range(num_classes)]
    if hidden_layers is None:
        model = {
            'W1': p['W1'], 'b1': p['b1'], 'W2': p['W2'], 'b2': p['b2'],
            'feature_dim': feature_dim, 'hidden_dim': hidden_dim,
            'num_classes': num_classes, 'accuracy': accuracy,
            'hidden_layers': None, 'class_names': IMAGE_CLASSES,
        }
    else:
        model = dict(p)
        model['feature_dim'] = feature_dim
        model['hidden_dim'] = hidden_dim
        model['num_classes'] = num_classes
        model['accuracy'] = accuracy
        model['hidden_layers'] = hidden_layers
        model['class_names'] = IMAGE_CLASSES

    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info("Image classifier sauvegarde dans %s (precision: %.1f%%)", save_path, accuracy)

    return {'model': model, 'save_path': save_path, 'accuracy': accuracy, 'history': history}


# ==================================================================
# train_speech_classifier
# ==================================================================

SPEECH_CLASSES = ["voyelle", "consonne", "silence"]


def train_speech_classifier(X=None, y=None, num_classes=3, feature_dim=26,
                            hidden_dim=64, lr=0.01, epochs=500,
                            save_path=None, seed=42, hidden_layers=None):
    """
    Entraîne un classifieur de phonèmes à partir de features audio.

    Architecture configurable en profondeur :

      - ``hidden_layers=None`` (défaut) : TemporalConv1D -> ReLU -> global avg pool
        -> FC -> softmax.  Comportement identique à l'original ; les paramètres FC
        sont sauvegardés sous les clés ``W_fc``, ``b_fc`` (rétrocompatible).

      - ``hidden_layers=[64, 32]`` : TemporalConv1D -> ReLU -> global avg pool
        -> FC(64) -> ReLU -> FC(32) -> ReLU -> FC(num_classes) -> softmax.
        Les paramètres FC sont sauvegardés sous les clés ``W_fc_0``, ``b_fc_0``,
        ``W_fc_1``, ``b_fc_1``, ..., ``W_fc_out``, ``b_fc_out``.

    Args:
        X: Features d'entrée (N, feature_dim).
        y: Labels (N,).
        num_classes: Nombre de classes (défaut : voyelle, consonne, silence).
        feature_dim: Dimension des features (défaut : 26).
        hidden_dim: Dimension de la convolution (nombre de filtres).
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        hidden_layers: Liste d'entiers décrivant les dimensions des couches FC
            cachées successives après le pooling. Si ``None``, une seule couche FC
            de dimension ``hidden_dim`` vers ``num_classes`` est utilisée.

    Returns:
        dict: Modèle et métriques.
    """
    ensure_directories()

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"speech_classifier{MODEL_EXTENSION}")

    if X is None or y is None:
        n_per_class = 30
        X = np.array(C.randn((num_classes * n_per_class, feature_dim), seed=seed)) * 0.1
        y = np.zeros(num_classes * n_per_class, dtype=int)
        for c in range(num_classes):
            start = c * n_per_class
            end = (c + 1) * n_per_class
            X[start:end, :13] += (c - 1) * 0.3
            y[start:end] = c

    # --- Initialisation convolution ---
    conv = TemporalConv1D(1, hidden_dim, kernel_size=3, dilation=1, seed=seed)
    conv_out_len = max(1, feature_dim - (3 - 1) * 1)

    # --- Initialisation couches FC avec profondeur variable ---
    if hidden_layers is None:
        fc_sizes = []  # direct: pooled -> num_classes
    else:
        fc_sizes = list(hidden_layers)
    n_fc = len(fc_sizes)

    fc_Ws = []
    fc_bs = []
    prev_dim = hidden_dim  # sortie du global avg pool
    for i, size in enumerate(fc_sizes):
        fc_Ws.append(xavier_init((prev_dim, size), seed=seed + 10 + i))
        fc_bs.append(np.array(C.zeros((1, size))))
        prev_dim = size
    fc_W_out = xavier_init((prev_dim, num_classes), seed=seed + 10 + n_fc)
    fc_b_out = np.array(C.zeros((1, num_classes)))

    best_loss = float('inf')
    best_params = None
    history = []

    for epoch in range(epochs):
        total_loss = 0
        indices = np.array(C.permutation(len(X), seed=seed + epoch))
        for idx in indices:
            xi = X[idx]
            yi = y[idx]

            # ---- Forward ----
            x_2d = xi.reshape(1, -1)  # (1, feature_dim)
            conv_out = conv.forward(x_2d)  # (hidden_dim, out_len)
            conv_act = relu(conv_out)  # (hidden_dim, out_len)
            pooled = np.array(C.mean_axis(conv_act, 1)).reshape(-1, 1)  # (hidden_dim, 1)
            pooled_t = pooled.T  # (1, hidden_dim)

            if hidden_layers is None:
                # Legacy : pooled -> FC(num_classes)
                z_final = C.matmul(pooled_t, fc_W_out) + fc_b_out
                out = softmax(z_final)[0]
            else:
                # Multi-FC : pooled -> FC(h1) -> ReLU -> FC(h2) -> ReLU -> ... -> FC(num_classes)
                z_list = []
                a_list = [pooled_t]
                a = pooled_t
                for i in range(n_fc):
                    z = C.matmul(a, fc_Ws[i]) + fc_bs[i]
                    z_list.append(z)
                    a = relu(z)
                    a_list.append(a)
                z_final = C.matmul(a, fc_W_out) + fc_b_out
                out = softmax(z_final)[0]

            loss = -np.log(float(out[yi]) + 1e-8)
            total_loss += loss

            # ---- Backward ----
            d_z = out.copy()
            d_z[yi] -= 1.0

            if hidden_layers is None:
                # Legacy backward : softmax -> FC -> pool -> conv
                d_z_fc = d_z.reshape(1, -1)
                d_fc_W_out = C.matmul(pooled_t.T, d_z_fc)
                d_fc_b_out = d_z_fc
                d_pooled_t = C.matmul(d_z_fc, fc_W_out.T)
            else:
                # Multi-FC backward
                d_z_final = d_z.reshape(1, -1)
                d_fc_W_out = C.matmul(a_list[-1].T, d_z_final)
                d_fc_b_out = d_z_final
                d_a = C.matmul(d_z_final, fc_W_out.T)

                d_fc_Ws = [None] * n_fc
                d_fc_bs = [None] * n_fc
                for i in range(n_fc - 1, -1, -1):
                    d_zi = d_a * relu_deriv(z_list[i])
                    d_fc_Ws[i] = C.matmul(a_list[i].T, d_zi)
                    d_fc_bs[i] = d_zi
                    d_a = C.matmul(d_zi, fc_Ws[i].T)

                d_pooled_t = d_a

            d_pooled = d_pooled_t.T  # (hidden_dim, 1)

            # Global avg pool backward
            d_conv_act = np.tile(d_pooled / conv_out_len, (1, conv_out_len))
            d_conv_out = d_conv_act * relu_deriv(conv_out)

            # Conv backward
            d_x_2d, d_W_conv, d_b_conv = conv.backward(d_conv_out)

            # ---- Mise à jour SGD ----
            fc_W_out += lr * d_fc_W_out
            fc_b_out += lr * d_fc_b_out
            if hidden_layers is not None:
                for i in range(n_fc):
                    fc_Ws[i] = fc_Ws[i] + lr * d_fc_Ws[i]
                    fc_bs[i] = fc_bs[i] + lr * d_fc_bs[i]
            conv.W += lr * d_W_conv
            conv.b += lr * d_b_conv

            # ---- Clip ----
            fc_W_out = C.clip(fc_W_out, -1, 1)
            if hidden_layers is not None:
                for i in range(n_fc):
                    fc_Ws[i] = C.clip(fc_Ws[i], -1, 1)
            conv.W = C.clip(conv.W, -1, 1)

        avg_loss = total_loss / len(X)
        history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            if hidden_layers is None:
                best_params = {
                    'conv_params': conv.get_params(),
                    'W_fc': fc_W_out.copy(), 'b_fc': fc_b_out.copy(),
                }
            else:
                best_params = {'conv_params': conv.get_params()}
                for i in range(n_fc):
                    best_params[f'W_fc_{i}'] = fc_Ws[i].copy()
                    best_params[f'b_fc_{i}'] = fc_bs[i].copy()
                best_params['W_fc_out'] = fc_W_out.copy()
                best_params['b_fc_out'] = fc_b_out.copy()

        if epoch % 100 == 0:
            logger.info("Epoch %d, Loss: %.6f", epoch, avg_loss)
        if avg_loss < 0.01:
            logger.info("Convergence a l'epoch %d", epoch)
            break

    # ---- Évaluation ----
    p = best_params
    best_conv = TemporalConv1D.from_params(p['conv_params'])
    correct = 0
    for i in range(len(X)):
        x_2d = X[i].reshape(1, -1)
        conv_out = best_conv.forward(x_2d)
        conv_act = relu(conv_out)
        pooled = np.array(C.mean_axis(conv_act, 1)).reshape(-1, 1).T  # (1, hidden_dim)
        if hidden_layers is None:
            z = C.matmul(pooled, p['W_fc']) + p['b_fc']
            out = softmax(z)[0]
        else:
            a = pooled
            for j in range(n_fc):
                z = C.matmul(a, p[f'W_fc_{j}']) + p[f'b_fc_{j}']
                a = relu(z)
            z = C.matmul(a, p['W_fc_out']) + p['b_fc_out']
            out = softmax(z)[0]
        pred = C.argmax(out)
        if pred == y[i]:
            correct += 1
    accuracy = correct / len(X) * 100

    # ---- Construction du dictionnaire modèle ----
    model = {
        'conv_params': p['conv_params'],
        'feature_dim': feature_dim, 'hidden_dim': hidden_dim,
        'num_classes': num_classes, 'class_names': SPEECH_CLASSES,
        'accuracy': accuracy,
        'hidden_layers': hidden_layers,
    }
    if hidden_layers is None:
        model['W_fc'] = p['W_fc']
        model['b_fc'] = p['b_fc']
    else:
        for i in range(n_fc):
            model[f'W_fc_{i}'] = p[f'W_fc_{i}']
            model[f'b_fc_{i}'] = p[f'b_fc_{i}']
        model['W_fc_out'] = p['W_fc_out']
        model['b_fc_out'] = p['b_fc_out']

    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info("Speech classifier sauvegarde dans %s (precision: %.1f%%)", save_path, accuracy)

    return {'model': model, 'save_path': save_path, 'accuracy': accuracy, 'history': history}