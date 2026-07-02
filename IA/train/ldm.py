"""
IA/train/ldm.py — Entraînement des modèles de diffusion (LDM).

Modèles :
  - train_ldm_image : Diffusion conditionnelle pour la génération d'images 2D
                       (formes géométriques).
  - train_ldm_audio : Diffusion conditionnelle pour la génération d'audio
                       (ondes sinusoïdales, carrées, etc.)
"""

import os
import math
import logging

import numpy as np

from ..config import MODELS_DIR, MODEL_EXTENSION, ensure_directories
from ..cpp import get_core
from ..ia_format import save_model, serialize_model_dict

C = get_core()

logger = logging.getLogger(__name__)


# ==================================================================
# Réseau de prédiction de bruit
# ==================================================================

class SimpleDiffusionNet:
    """
    Réseau pour prédire le bruit ajouté lors de la diffusion, avec profondeur
    configurable via ``hidden_sizes``.

    Architecture (hidden_sizes=None, backward compatible) :
        Embedding classe (num_classes → hidden_dim)
        FC1 : (input_dim + hidden_dim) → hidden_dim, ReLU
        FC2 : hidden_dim → input_dim

    Architecture (hidden_sizes=[h0, h1, ...]) :
        Embedding classe (num_classes → h0)
        FC_0 : (input_dim + h0) → h0, ReLU
        FC_1 : h0 → h1, ReLU
        …
        FC_N : h_{N-1} → input_dim   (pas de ReLU sur la dernière couche)

    Args:
        input_dim: Dimension de l'entrée (et de la sortie).
        num_classes: Nombre de classes conditionnelles.
        hidden_dim: Dimension cachée utilisée quand *hidden_sizes* est ``None``.
        hidden_sizes: Liste optionnelle de tailles cachées.  ``None`` produit
            le réseau historique à 2 couches ; une liste ``[128, 64]`` crée
            3 couches FC (input+embed → 128 → 64 → input_dim).
    """

    def __init__(self, input_dim, num_classes, hidden_dim=128, hidden_sizes=None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.hidden_sizes = hidden_sizes

        # Déterminer embed_dim et les tailles des couches FC
        if hidden_sizes is None:
            # Comportement historique : réseau à 2 couches
            self.embed_dim = hidden_dim
            fc_out_sizes = [hidden_dim, input_dim]
        else:
            self.embed_dim = hidden_sizes[0]
            fc_out_sizes = list(hidden_sizes) + [input_dim]

        # Embedding de classe
        self.class_embedding = np.random.randn(num_classes, self.embed_dim) * 0.01

        # Construction des couches FC : chaque couche stocke W et b
        self.fc_layers = []
        prev_size = input_dim + self.embed_dim
        for out_size in fc_out_sizes:
            W = (np.random.randn(prev_size, out_size)
                 * np.sqrt(2.0 / (prev_size + out_size)))
            b = np.zeros(out_size)
            self.fc_layers.append({'W': W, 'b': b})
            prev_size = out_size

    def predict_noise(self, x_noisy, class_id):
        """
        Prédit le bruit à partir de l'entrée bruitée et de la classe.

        Args:
            x_noisy: ndarray de forme (input_dim,) ou (1, input_dim).
            class_id: int, identifiant de classe.

        Returns:
            tuple: (predicted_noise, x_concat, z_list, layer_inputs)
                - predicted_noise: (1, input_dim)
                - x_concat: (1, input_dim + embed_dim), concaténation entrée + embed
                - z_list: liste des pré-activations [z_0, z_1, …] pour chaque FC
                - layer_inputs: liste des entrées de chaque FC [a_0, a_1, …]
                  (a_0 = x_concat, a_i = relu(z_{i-1}) pour i > 0)
        """
        x_noisy = np.asarray(x_noisy, dtype=np.float64)
        if x_noisy.ndim == 1:
            x_noisy = x_noisy.reshape(1, -1)
        c_embed = self.class_embedding[class_id].reshape(1, -1)
        x_concat = np.concatenate([x_noisy, c_embed], axis=1)  # (1, input_dim+embed_dim)

        z_list = []
        layer_inputs = []

        N = len(self.fc_layers)
        a = x_concat
        for i in range(N):
            layer_inputs.append(a)
            z = C.matmul(a, self.fc_layers[i]['W']) + self.fc_layers[i]['b'].reshape(1, -1)
            z_list.append(z)
            if i < N - 1:
                a = C.relu(z)
            else:
                a = z  # pas de ReLU sur la dernière couche
        output = a

        return output, x_concat, z_list, layer_inputs

    def get_params(self):
        """
        Retourne un dictionnaire de tous les paramètres.

        Les poids sont toujours sauvegardés avec les clés indexées
        ``W_0``, ``b_0``, ``W_1``, ``b_1``, …
        """
        params = {
            'class_embedding': self.class_embedding,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'hidden_sizes': self.hidden_sizes,
        }

        for i, layer in enumerate(self.fc_layers):
            params[f'W_{i}'] = layer['W']
            params[f'b_{i}'] = layer['b']

        return params


# ==================================================================
# Diffusion avant (ajout de bruit)
# ==================================================================

def _add_noise(x_0, t, alpha_cumprod, noise):
    """Ajoute du bruit gaussien au signal x_0 au pas de temps t."""
    sqrt_alpha = math.sqrt(alpha_cumprod[t])
    sqrt_one_minus = math.sqrt(1.0 - alpha_cumprod[t])
    return sqrt_alpha * x_0 + sqrt_one_minus * noise


# ==================================================================
# Génération de données synthétiques — images
# ==================================================================

def _generate_image_data(image_size=8, num_classes=5, num_samples_per_class=20, seed=42):
    """
    Génère des images 2D synthétiques pour *num_classes* classes.

    Chaque classe est identifiée par son index (0, 1, ..., N-1).
    Le motif de chaque classe est déterminé paramétriquement à partir
    de son index, sans noms fixés.  *num_classes* accepte tout entier >= 1.
    Valeurs normalisées dans [-1, 1].
    """
    num_classes = max(1, num_classes)
    rng = np.random.RandomState(seed)
    images = []
    labels = []

    # Pool de générateurs de formes (utilisé cycliquement)
    def _shape_0(r):
        img = np.full((image_size, image_size), -1.0)
        w = r.randint(2, image_size - 1)
        h = r.randint(2, image_size - 1)
        x0 = r.randint(0, max(1, image_size - w))
        y0 = r.randint(0, max(1, image_size - h))
        img[y0:y0 + h, x0:x0 + w] = 1.0
        return img

    def _shape_1(r):
        img = np.full((image_size, image_size), -1.0)
        cy = r.uniform(1.5, image_size - 2.5)
        cx = r.uniform(1.5, image_size - 2.5)
        rad = r.uniform(1.0, image_size / 2.0 - 0.5)
        for iy in range(image_size):
            for ix in range(image_size):
                if (iy - cy) ** 2 + (ix - cx) ** 2 <= rad ** 2:
                    img[iy, ix] = 1.0
        return img

    def _shape_2(r):
        img = np.full((image_size, image_size), -1.0)
        apex_y = r.randint(0, image_size // 3)
        apex_x = r.randint(image_size // 3, 2 * image_size // 3)
        base_y = r.randint(2 * image_size // 3, image_size)
        left_x = r.randint(0, image_size // 3)
        right_x = r.randint(2 * image_size // 3, image_size)
        for iy in range(apex_y, base_y + 1):
            if iy <= base_y:
                progress = (iy - apex_y) / max(1, base_y - apex_y)
                lx = int(apex_x + progress * (left_x - apex_x))
                rx = int(apex_x + progress * (right_x - apex_x))
                lx = max(0, min(lx, image_size - 1))
                rx = max(0, min(rx, image_size - 1))
                img[iy, lx:rx + 1] = 1.0
        return img

    def _shape_3(r):
        img = np.full((image_size, image_size), -1.0)
        thickness = r.randint(1, max(2, image_size // 4))
        cy = image_size // 2
        cx = image_size // 2
        arm_len = r.randint(image_size // 3, image_size // 2 + 1)
        y_start = max(0, cy - arm_len)
        y_end = min(image_size, cy + arm_len + 1)
        x_start = max(0, cx - arm_len)
        x_end = min(image_size, cx + arm_len + 1)
        half_t = thickness // 2
        img[y_start:y_end, cx - half_t:cx - half_t + thickness] = 1.0
        img[cy - half_t:cy - half_t + thickness, x_start:x_end] = 1.0
        return img

    def _shape_4(r):
        img = np.full((image_size, image_size), -1.0)
        cy = r.uniform(2.0, image_size - 3.0)
        cx = r.uniform(2.0, image_size - 3.0)
        ry = r.uniform(1.0, image_size / 2.0 - 0.5)
        rx = r.uniform(1.0, image_size / 2.0 - 0.5)
        for iy in range(image_size):
            for ix in range(image_size):
                if ((iy - cy) / ry) ** 2 + ((ix - cx) / rx) ** 2 <= 1.0:
                    img[iy, ix] = 1.0
        return img

    def _shape_5(r):
        img = np.full((image_size, image_size), -1.0)
        cy = image_size / 2.0
        cx = image_size / 2.0
        hw = r.uniform(image_size / 4.0, image_size / 2.0 - 0.5)
        hh = r.uniform(image_size / 4.0, image_size / 2.0 - 0.5)
        for iy in range(image_size):
            for ix in range(image_size):
                if abs(ix - cx) / hw + abs(iy - cy) / hh <= 1.0:
                    img[iy, ix] = 1.0
        return img

    def _shape_6(r):
        img = np.full((image_size, image_size), -1.0)
        y0 = r.randint(0, max(1, image_size // 2))
        h = r.randint(1, max(2, image_size // 3))
        y0 = min(y0, image_size - h)
        img[y0:y0 + h, :] = 1.0
        return img

    def _shape_7(r):
        img = np.full((image_size, image_size), -1.0)
        x0 = r.randint(0, max(1, image_size // 2))
        w = r.randint(1, max(2, image_size // 3))
        x0 = min(x0, image_size - w)
        img[:, x0:x0 + w] = 1.0
        return img

    def _shape_8(r):
        img = np.full((image_size, image_size), -1.0)
        x0 = r.randint(0, max(1, image_size // 2))
        w = r.randint(1, max(2, image_size // 3))
        x0 = min(x0, image_size - w)
        y0 = r.randint(0, max(1, image_size // 2))
        h = r.randint(1, max(2, image_size // 3))
        y0 = min(y0, image_size - h)
        img[y0:y0 + h, x0:x0 + w] = -1.0
        return img

    def _shape_9(r):
        img = np.full((image_size, image_size), -1.0)
        cy = image_size / 2.0
        cx = image_size / 2.0
        rad = r.uniform(image_size / 4.0, image_size / 2.0 - 0.5)
        for iy in range(image_size):
            for ix in range(image_size):
                if (iy - cy) ** 2 + (ix - cx) ** 2 <= rad ** 2:
                    img[iy, ix] = -1.0
        return img

    _pool = [_shape_0, _shape_1, _shape_2, _shape_3, _shape_4,
             _shape_5, _shape_6, _shape_7, _shape_8, _shape_9]

    def _gen_random_pattern(class_id):
        local_rng = np.random.RandomState(seed + 1000 + class_id)
        img = np.full((image_size, image_size), -1.0)
        density = 0.2 + 0.4 * ((class_id * 7 + 3) % 10) / 10.0
        mask = local_rng.rand(image_size, image_size) < density
        img[mask] = 1.0
        return img

    for cls in range(num_classes):
        if cls < len(_pool):
            gen = _pool[cls]
        else:
            gen = lambda r, c=cls: _gen_random_pattern(c)
        for _ in range(num_samples_per_class):
            img = gen(rng)
            images.append(img.flatten())
            labels.append(cls)

    return np.array(images, dtype=np.float64), np.array(labels, dtype=np.int64)


# ==================================================================
# Génération de données synthétiques — audio
# ==================================================================

def _generate_audio_data(signal_length=64, num_classes=5, num_samples_per_class=20, seed=42):
    """
    Génère des signaux audio synthétiques pour *num_classes* classes.

    Chaque classe est identifiée par son index (0, 1, ..., N-1).
    Le signal de chaque classe est déterminé paramétriquement à partir
    de son index, sans noms fixés.  *num_classes* accepte tout entier >= 1.
    Valeurs normalisées dans [-1, 1].
    """
    num_classes = max(1, num_classes)
    rng = np.random.RandomState(seed)
    signals = []
    labels = []
    t = C.linspace(0, 1, signal_length)

    # Pool de générateurs de signaux (utilisé cycliquement)
    def _wave_0(r):
        freq = r.uniform(1, 10)
        phase = r.uniform(0, 2 * np.pi)
        return np.sin(2 * np.pi * freq * t + phase)

    def _wave_1(r):
        freq = r.uniform(1, 10)
        phase = r.uniform(0, 2 * np.pi)
        return np.sign(np.sin(2 * np.pi * freq * t + phase))

    def _wave_2(r):
        freq = r.uniform(1, 10)
        return 2.0 * (t * freq - np.floor(0.5 + t * freq))

    def _wave_3(r):
        signal = r.randn(signal_length) * 0.5
        return np.clip(signal, -1.0, 1.0)

    def _wave_4(r):
        f0 = r.uniform(1, 5)
        f1 = r.uniform(5, 15)
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / 2.0)
        return np.sin(phase)

    def _wave_5(r):
        freq = r.uniform(1, 10)
        return 2.0 * np.abs(2.0 * (t * freq - np.floor(t * freq + 0.5))) - 1.0

    def _wave_6(r):
        f_carrier = r.uniform(5, 15)
        f_mod = r.uniform(0.5, 3)
        m_depth = r.uniform(0.3, 1.0)
        signal = (1.0 + m_depth * np.sin(2 * np.pi * f_mod * t)) * np.sin(2 * np.pi * f_carrier * t)
        return np.clip(signal, -1.0, 1.0)

    def _wave_7(r):
        freq = r.uniform(1, 10)
        return np.cos(2 * np.pi * freq * t)

    def _wave_8(r):
        f0 = r.uniform(5, 10)
        f1 = r.uniform(1, 5)
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / 2.0)
        return np.sin(phase)

    def _wave_9(r):
        freq = r.uniform(1, 8)
        signal = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(4 * np.pi * freq * t)
        mx = np.max(np.abs(signal))
        if mx > 0:
            signal /= mx
        return signal

    _pool = [_wave_0, _wave_1, _wave_2, _wave_3, _wave_4,
             _wave_5, _wave_6, _wave_7, _wave_8, _wave_9]

    def _gen_harmonic_mix(class_id):
        local_rng = np.random.RandomState(seed + 1000 + class_id)
        n_harmonics = 2 + class_id % 5
        signal = np.zeros(signal_length)
        for h in range(1, n_harmonics + 1):
            freq = local_rng.uniform(1, 12)
            amp = 1.0 / h
            phase = local_rng.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * freq * h * t + phase)
        mx = np.max(np.abs(signal))
        if mx > 0:
            signal /= mx
        return signal

    for cls in range(num_classes):
        if cls < len(_pool):
            gen = _pool[cls]
        else:
            gen = lambda r, c=cls: _gen_harmonic_mix(c)
        for _ in range(num_samples_per_class):
            signal = gen(rng)
            signals.append(signal)
            labels.append(cls)

    return np.array(signals, dtype=np.float64), np.array(labels, dtype=np.int64)


# ==================================================================
# Boucle d'entraînement diffusion partagée
# ==================================================================

def _train_ldm_core(X, class_ids, input_dim, num_classes, timesteps, lr,
                    epochs, save_path, seed, filename, hidden_dim=128,
                    hidden_sizes=None):
    """
    Boucle d'entraînement LDM partagée.

    Utilise un schedule beta linéaire de 1e-4 à 0.02.
    Le réseau prédit le bruit ajouté lors de la diffusion avant.

    Args:
        X: Données d'entraînement (n_samples, input_dim).
        class_ids: Labels de classe (n_samples,).
        input_dim: Dimension de l'entrée.
        num_classes: Nombre de classes.
        timesteps: Nombre de pas de diffusion.
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        save_path: Chemin de sauvegarde (ou None pour le chemin par défaut).
        seed: Graine aléatoire.
        filename: Nom de base du fichier de sauvegarde.
        hidden_dim: Dimension cachée (utilisée quand *hidden_sizes* est None).
        hidden_sizes: Liste optionnelle de tailles cachées pour contrôler la
            profondeur du réseau.  ``None`` → réseau à 2 couches historique.
            ``[128, 64]`` → 3 couches FC, etc.
    """
    ensure_directories()
    np.random.seed(seed)

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"{filename}{MODEL_EXTENSION}")

    # --- Schedule de diffusion ---
    betas = np.linspace(1e-4, 0.02, timesteps)
    alphas = 1.0 - betas
    alpha_cumprod = np.cumprod(alphas)

    # --- Modèle ---
    model = SimpleDiffusionNet(input_dim, num_classes, hidden_dim, hidden_sizes)

    n_samples = len(X)
    N = len(model.fc_layers)
    history_loss = []
    log_interval = max(1, epochs // 10)

    _seed_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        # Mélanger les indices
        indices = C.permutation(n_samples, seed + epoch)

        for idx in indices:
            x_0 = X[idx]
            cid = class_ids[idx]

            # Tirer un pas de temps aléatoire
            t = np.random.randint(0, timesteps)
            noise = C.randn((input_dim,), seed + _seed_counter)
            _seed_counter += 1
            x_noisy = _add_noise(x_0, t, alpha_cumprod, noise)

            # Passe avant
            predicted, x_concat, z_list, layer_inputs = model.predict_noise(x_noisy, cid)

            # Perte MSE (predicted et noise doivent avoir la même shape)
            noise_2d = noise.reshape(1, -1) if noise.ndim == 1 else noise
            loss = C.mse_loss(predicted, noise_2d)
            epoch_loss += loss

            # --- Rétropropagation ---
            pred_flat = predicted.flatten()
            noise_flat = noise.flatten()
            d = (2.0 * (pred_flat - noise_flat) / input_dim).reshape(1, -1)  # (1, output_dim)

            # Calculer les gradients pour toutes les couches (de la dernière à la première)
            grads_W = [None] * N
            grads_b = [None] * N

            for i in range(N - 1, -1, -1):
                # d est le gradient w.r.t. z_i (pré-activation de la couche i)
                grads_W[i] = C.outer(layer_inputs[i].flatten(), d.flatten())
                grads_b[i] = d.flatten().copy()

                if i > 0:
                    # Rétropropager à travers la couche i
                    d = C.matmul(d, model.fc_layers[i]['W'].T)
                    # À travers ReLU après la couche i-1
                    d = C.mul(d, C.relu_deriv(z_list[i - 1]))

            # Gradient de l'embedding de classe
            # d contient d_z_0 ; on rétropropage à travers la première couche
            d_x_concat = C.matmul(d, model.fc_layers[0]['W'].T)  # (1, input_dim+embed_dim)
            d_class_embedding = np.zeros_like(model.class_embedding)
            d_class_embedding[cid] = d_x_concat.flatten()[input_dim:]

            # --- Mise à jour (tous les gradients calculés avant la mise à jour) ---
            for i in range(N):
                model.fc_layers[i]['W'] -= lr * grads_W[i]
                model.fc_layers[i]['b'] -= lr * grads_b[i]
            model.class_embedding -= lr * d_class_embedding

        avg_loss = epoch_loss / n_samples
        history_loss.append(float(avg_loss))

        if epoch % log_interval == 0:
            logger.info("Epoch %d/%d  loss: %.6f", epoch, epochs, avg_loss)

    # --- Évaluation finale : MSE moyen sur tout le jeu ---
    total_mse = 0.0
    for i in range(n_samples):
        t_eval = np.random.randint(0, timesteps)
        noise_eval = C.randn((input_dim,), seed + _seed_counter)
        _seed_counter += 1
        x_noisy_eval = _add_noise(X[i], t_eval, alpha_cumprod, noise_eval)
        pred_eval, _, _, _ = model.predict_noise(x_noisy_eval, class_ids[i])
        noise_eval_2d = noise_eval.reshape(1, -1) if noise_eval.ndim == 1 else noise_eval
        total_mse += C.mse_loss(pred_eval, noise_eval_2d)
    final_mse = total_mse / n_samples
    accuracy = float(max(0.0, 1.0 - final_mse))

    # --- Sauvegarde ---
    saved_model = model.get_params()
    saved_model['betas'] = betas
    saved_model['alpha_cumprod'] = alpha_cumprod
    saved_model['timesteps'] = timesteps

    header, tensors = serialize_model_dict(saved_model)
    save_model(save_path, header, tensors)

    logger.info(
        "%s sauvegardé dans %s (accuracy: %.4f, MSE final: %.6f)",
        filename, save_path, accuracy, final_mse,
    )

    history = {'loss': history_loss}

    return {
        'model': saved_model,
        'save_path': save_path,
        'accuracy': accuracy,
        'history': history,
    }


# ==================================================================
# LDM Image
# ==================================================================

def train_ldm_image(image_size=8, num_classes=5, timesteps=200, lr=0.001,
                    epochs=1000, save_path=None, seed=42, hidden_sizes=None):
    """
    Entraîne un modèle de diffusion conditionnelle pour la génération
    d'images 2D.

    Chaque classe est identifiée par son index (0, 1, ..., N-1).
    Le nombre de classes est paramétrable via *num_classes*.

    Args:
        image_size: Taille des images carrées (H = W).
        num_classes: Nombre de classes conditionnelles.
        timesteps: Nombre de pas de diffusion.
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        hidden_sizes: Liste optionnelle de tailles cachées pour contrôler la
            profondeur du réseau.  ``None`` → réseau à 2 couches (défaut).
            ``[128, 64]`` → 3 couches FC, etc.

    Returns:
        dict: {'model', 'save_path', 'accuracy', 'history'}
    """
    X, class_ids = _generate_image_data(
        image_size=image_size, num_classes=num_classes,
        num_samples_per_class=20, seed=seed,
    )
    input_dim = image_size * image_size

    return _train_ldm_core(
        X=X,
        class_ids=class_ids,
        input_dim=input_dim,
        num_classes=num_classes,
        timesteps=timesteps,
        lr=lr,
        epochs=epochs,
        save_path=save_path,
        seed=seed,
        filename="ldm_image",
        hidden_dim=128,
        hidden_sizes=hidden_sizes,
    )


# ==================================================================
# LDM Audio
# ==================================================================

def train_ldm_audio(signal_length=64, num_classes=5, timesteps=200, lr=0.001,
                    epochs=1000, save_path=None, seed=42, hidden_sizes=None):
    """
    Entraîne un modèle de diffusion conditionnelle pour la génération
    de signaux audio.

    Chaque classe est identifiée par son index (0, 1, ..., N-1).
    Le nombre de classes est paramétrable via *num_classes*.

    Args:
        signal_length: Longueur du signal.
        num_classes: Nombre de classes conditionnelles.
        timesteps: Nombre de pas de diffusion.
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        hidden_sizes: Liste optionnelle de tailles cachées pour contrôler la
            profondeur du réseau.  ``None`` → réseau à 2 couches (défaut).
            ``[128, 64]`` → 3 couches FC, etc.

    Returns:
        dict: {'model', 'save_path', 'accuracy', 'history'}
    """
    X, class_ids = _generate_audio_data(
        signal_length=signal_length, num_classes=num_classes,
        num_samples_per_class=20, seed=seed,
    )
    input_dim = signal_length

    return _train_ldm_core(
        X=X,
        class_ids=class_ids,
        input_dim=input_dim,
        num_classes=num_classes,
        timesteps=timesteps,
        lr=lr,
        epochs=epochs,
        save_path=save_path,
        seed=seed,
        filename="ldm_audio",
        hidden_dim=128,
        hidden_sizes=hidden_sizes,
    )