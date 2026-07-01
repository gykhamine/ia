"""
IA/infer/vision.py — Inférence des modèles de classification perceptionnelle.
"""

import pickle
import logging

import numpy as np

from ..cpp import get_core
C = get_core()

logger = logging.getLogger(__name__)


# ==================================================================
# Helpers
# ==================================================================

def relu(x):
    """Fonction d'activation ReLU."""
    return C.relu(x)


def softmax(x):
    """Softmax stable le long du dernier axe."""
    return C.softmax(x)


# ==================================================================
# Extracteurs de caractéristiques (identiques au module d'entraînement)
# ==================================================================

class SimpleFeatureExtractor:
    """Extraction de caractéristiques statistiques depuis une image 2D/3D."""

    def __init__(self, num_features=16):
        self.num_features = num_features

    def extract(self, image):
        """
        Extrait un vecteur de caractéristiques à partir d'une image.

        Args:
            image: Tableau numpy 2D ou 3D.

        Returns:
            ndarray: Vecteur de caractéristiques de taille num_features.
        """
        # Convertir en 2D gris si nécessaire
        if image.ndim == 3:
            img = np.mean(image.astype(float), axis=2)
        else:
            img = image.astype(float)

        flat = img.flatten()
        features = np.zeros(self.num_features)

        features[0] = np.mean(flat)
        features[1] = np.std(flat)
        features[2] = np.max(flat)
        features[3] = np.min(flat)
        features[4] = np.median(flat)
        features[5] = np.sum(flat > np.mean(flat)) / max(flat.size, 1)

        if flat.size > 1:
            features[6] = np.percentile(flat, 25)
            features[7] = np.percentile(flat, 75)
        else:
            features[6] = flat[0]
            features[7] = flat[0]

        # Indices spatiaux (utilise img 2D, pas flat)
        if img.ndim >= 2:
            rows, cols = img.shape[0], img.shape[1]
            center_r, center_c = rows / 2, cols / 2
            rr, cc = np.mgrid[0:rows, 0:cols]
            dist = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)
            features[8] = np.mean(img * dist) / (np.max(dist) + 1e-8)

            # Symétries
            if rows > 1:
                features[9] = np.mean(np.abs(img - img[::-1]))
            else:
                features[9] = 0.0
            if cols > 1:
                features[10] = np.mean(np.abs(img - img[:, ::-1]))
            else:
                features[10] = 0.0

            # Gradient moyen (approximation)
            if rows > 1 and cols > 1:
                grad_r = np.diff(img, axis=0)
                grad_c = np.diff(img, axis=1)
                features[11] = np.mean(np.abs(grad_r))
                features[12] = np.mean(np.abs(grad_c))
            else:
                features[11] = 0.0
                features[12] = 0.0
        else:
            features[8:13] = 0.0

        # Moments d'ordre supérieur
        features[13] = float(np.mean((flat - np.mean(flat)) ** 3))
        features[14] = float(np.mean((flat - np.mean(flat)) ** 4))
        features[15] = float(np.sum(flat ** 2))

        return features


class AudioFeatureExtractor:
    """Extraction de caractéristiques depuis un signal audio 1D."""

    def __init__(self, num_features=16):
        self.num_features = num_features

    def extract(self, signal):
        """
        Extrait un vecteur de caractéristiques à partir d'un signal audio.

        Args:
            signal: Tableau numpy 1D.

        Returns:
            ndarray: Vecteur de caractéristiques de taille num_features.
        """
        features = np.zeros(self.num_features)
        n = len(signal)

        features[0] = np.mean(signal)
        features[1] = np.std(signal)
        features[2] = np.max(signal)
        features[3] = np.min(signal)
        features[4] = np.median(signal)

        # Énergie
        features[5] = np.sum(signal ** 2) / n

        # ZCR (Zero Crossing Rate)
        if n > 1:
            zcr = np.sum(np.abs(np.diff(np.sign(signal)))) / (2 * n)
            features[6] = zcr
        else:
            features[6] = 0.0

        # Plage dynamique
        features[7] = features[2] - features[3]

        # Différences premières
        if n > 1:
            diff = np.diff(signal)
            features[8] = np.mean(np.abs(diff))
            features[9] = np.std(diff)
        else:
            features[8] = 0.0
            features[9] = 0.0

        # Énergie dans différentes tranches
        if n >= 4:
            quarter = n // 4
            features[10] = np.sum(signal[:quarter] ** 2) / quarter
            features[11] = np.sum(signal[quarter:2*quarter] ** 2) / quarter
            features[12] = np.sum(signal[2*quarter:3*quarter] ** 2) / quarter
        else:
            features[10:13] = features[5]

        # Spectre simplifié (magnitude DFT)
        if n > 1:
            spectrum = np.abs(np.fft.rfft(signal))
            features[13] = np.max(spectrum) if len(spectrum) > 0 else 0.0
            features[14] = np.mean(spectrum) if len(spectrum) > 0 else 0.0
            # Fréquence dominante (indice)
            features[15] = float(np.argmax(spectrum)) if len(spectrum) > 0 else 0.0
        else:
            features[13:16] = 0.0

        return features


# ==================================================================
# Classifieur d'images
# ==================================================================

def load_image_classifier(path):
    """
    Charge un modèle de classification d'images depuis un fichier pickle.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du modèle (W1, b1, W2, b2, class_names, etc.).
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Classifieur d'images chargé depuis %s", path)
    return model


def _is_new_format_image_classifier(model):
    """Détecte le nouveau format multi-couches (hidden_layers != None)."""
    return model.get('hidden_layers') is not None


def _predict_image_classifier_new(model, features):
    """Forward pass multi-couches pour le classifieur d'images."""
    hidden_layers = model['hidden_layers']
    n_hidden = len(hidden_layers)
    a = features.reshape(1, -1) if features.ndim == 1 else features
    for i in range(n_hidden):
        z = C.matmul(a, model[f'W_{i}']) + model[f'b_{i}']
        a = C.relu(z)
    logits = C.matmul(a, model['W_out']) + model['b_out']
    return logits


def _predict_image_classifier_legacy(model, features):
    """Forward pass historique 2 couches pour le classifieur d'images."""
    a = features.reshape(1, -1) if features.ndim == 1 else features
    h = C.relu(C.matmul(a, model['W1']) + model['b1'])
    if 'W2' in model:
        logits = C.matmul(h, model['W2']) + model['b2']
    else:
        logits = C.matmul(h, model['W_cls']) + model['b_cls']
    return logits


def predict_image(model, image):
    """
    Classification d'une image 2D/3D par extraction de caractéristiques
    et réseau entièrement connecté.

    Supporte deux formats de modèle :
      - Legacy (hidden_layers=None) : clés W1, b1, W2, b2
      - Nouveau (hidden_layers=[...]) : clés W_0, b_0, …, W_out, b_out

    Args:
        model: Dictionnaire de paramètres (issu de load_image_classifier).
        image: Tableau numpy 2D ou 3D.

    Returns:
        dict: {
            'class': str,
            'confidence': float,
            'probabilities': dict,
            'feature_vector': ndarray,
        }
    """
    # Déterminer la dimension d'entrée pour l'extracteur
    if _is_new_format_image_classifier(model):
        input_dim = model.get('feature_dim', 128)
    else:
        input_dim = model['W1'].shape[0]

    extractor = SimpleFeatureExtractor(
        num_features=model.get('num_features', input_dim)
    )
    features = extractor.extract(image)
    feature_vector = features.copy()

    # Forward pass selon le format
    if _is_new_format_image_classifier(model):
        logits = _predict_image_classifier_new(model, features)
    else:
        logits = _predict_image_classifier_legacy(model, features)

    probs = np.asarray(softmax(logits)).flatten()
    num_classes = len(probs)
    class_names = model.get('class_names', [f'Classe {i}' for i in range(num_classes)])
    sorted_indices = np.argsort(-probs)

    top_idx = int(sorted_indices[0])
    top_class = class_names[top_idx] if top_idx < len(class_names) else f'Classe {top_idx}'
    top_prob = float(probs[top_idx])
    confidence = top_prob

    probabilities = {}
    for idx in range(min(num_classes, len(class_names))):
        name = class_names[idx] if idx < len(class_names) else f'Classe {idx}'
        probabilities[name] = float(probs[idx])

    logger.info("Image — classe: %s, confiance: %.4f", top_class, confidence)

    return {
        'class': top_class,
        'confidence': confidence,
        'probabilities': probabilities,
        'feature_vector': feature_vector,
    }


# ==================================================================
# Classifieur de parole
# ==================================================================

def load_speech_classifier(path):
    """
    Charge un modèle de classification de parole depuis un fichier pickle.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du modèle.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Classifieur de parole chargé depuis %s", path)
    return model


def _is_new_format_speech_classifier(model):
    """Détecte le nouveau format multi-couches FC (hidden_layers != None)."""
    return model.get('hidden_layers') is not None


def _speech_conv_forward(model, x_2d):
    """Passe avant de la convolution 1D temporelle du speech classifier.

    Utilise conv_params (format de train/vision.py TemporalConv1D).
    """
    conv_p = model['conv_params']
    W = conv_p['W']       # (out_channels, in_channels, kernel_size)
    b = conv_p['b']       # (out_channels,)
    kernel_size = conv_p['kernel_size']
    dilation = conv_p.get('dilation', 1)
    out_channels = conv_p['out_channels']

    seq_len = x_2d.shape[1]
    out_len = max(1, seq_len - (kernel_size - 1) * dilation)

    conv_out = np.zeros((out_channels, out_len))
    for co in range(out_channels):
        for j in range(out_len):
            total = 0.0
            for ki in range(kernel_size):
                idx = j + ki * dilation
                if idx < seq_len:
                    total += x_2d[0, idx] * W[co, 0, ki]
            conv_out[co, j] = total + float(b[co])

    # ReLU + global avg pool -> (1, out_channels)
    conv_act = np.maximum(0.0, conv_out)
    pooled = conv_act.mean(axis=1).reshape(1, -1)
    return pooled


def predict_speech(model, audio_signal):
    """
    Classification d'un signal audio 1D par extraction de caractéristiques,
    convolution temporelle et réseau FC avec profondeur variable.

    Supporte deux formats de modèle :
      - Legacy (hidden_layers=None) : clés conv_params, W_fc, b_fc
      - Nouveau (hidden_layers=[...]) : clés conv_params, W_fc_0, b_fc_0, …, W_fc_out, b_fc_out

    Args:
        model: Dictionnaire de paramètres (issu de load_speech_classifier).
        audio_signal: Tableau numpy 1D.

    Returns:
        dict: {
            'class': str,
            'confidence': float,
            'probabilities': dict,
            'feature_vector': ndarray,
        }
    """
    extractor = AudioFeatureExtractor(
        num_features=model.get('num_features', 16)
    )
    features = extractor.extract(audio_signal)
    feature_vector = features.copy()

    x_2d = features.reshape(1, -1)

    if 'conv_params' in model:
        pooled = _speech_conv_forward(model, x_2d)
    else:
        pooled = x_2d

    # Réseau FC
    if _is_new_format_speech_classifier(model):
        hidden_layers = model['hidden_layers']
        n_fc = len(hidden_layers)
        a = pooled
        for i in range(n_fc):
            z = C.matmul(a, model[f'W_fc_{i}']) + model[f'b_fc_{i}']
            a = C.relu(z)
        logits = C.matmul(a, model['W_fc_out']) + model['b_fc_out']
    elif 'W_fc' in model:
        logits = C.matmul(pooled, model['W_fc']) + model['b_fc']
    elif 'W1' in model:
        h = C.relu(C.matmul(pooled, model['W1']) + model['b1'])
        if 'W2' in model:
            logits = C.matmul(h, model['W2']) + model['b2']
        else:
            logits = C.matmul(h, model['W_cls']) + model['b_cls']
    else:
        logits = C.matmul(pooled, model['W_cls']) + model['b_cls']

    probs = np.asarray(softmax(logits)).flatten()
    class_names = model['class_names']
    sorted_indices = np.argsort(-probs)

    top_idx = int(sorted_indices[0])
    top_class = class_names[top_idx] if top_idx < len(class_names) else f'Classe {top_idx}'
    top_prob = float(probs[top_idx])
    confidence = top_prob

    probabilities = {}
    for idx in range(len(class_names)):
        name = class_names[idx] if idx < len(class_names) else f'Classe {idx}'
        probabilities[name] = float(probs[idx])

    logger.info("Parole — classe: %s, confiance: %.4f", top_class, confidence)

    return {
        'class': top_class,
        'confidence': confidence,
        'probabilities': probabilities,
        'feature_vector': feature_vector,
    }