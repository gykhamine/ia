"""
IA/train/gan.py — Entraînement des GAN.

Modèles :
  - train_gan_1d : GAN 1D, génère une valeur scalaire (mélange de 2 gaussiennes).
  - train_gan_nd : GAN N-D, génère des vecteurs de dimension DATA_DIM.
  - train_gan_3d : GAN 3D, génère des volumes 4x4x4 (64 voxels).
  - train_gan_rgb : GAN RGB, génère des images 32x32x3 (3072 pixels).

Architecture configurable via generator_layers et discriminator_layers.
Si None, architecture par défaut à 3 couches (comportement identique
à la version originale pour la rétrocompatibilité).
"""

import os
import logging
import pickle
import random

import numpy as np

from ..config import MODELS_DIR, MODEL_EXTENSION, ensure_directories
from ..cpp import get_core

C = get_core()

logger = logging.getLogger(__name__)

# Seed counter for deterministic C++ engine calls
_seed_counter = 0


def _next_seed():
    """Return the next seed value for C++ engine calls."""
    global _seed_counter
    _seed_counter += 1
    return _seed_counter


# ==================================================================
# Fonctions utilitaires
# ==================================================================

def xavier_init(shape):
    """Initialisation Xavier."""
    return C.xavier_init(tuple(shape), _next_seed())


def sigmoid(x):
    """Fonction d'activation sigmoïde."""
    return C.sigmoid(x)


def leaky_relu(x, alpha=0.01):
    """Fonction d'activation Leaky ReLU."""
    return C.leaky_relu(x, alpha)


def leaky_relu_deriv(x, alpha=0.01):
    """Dérivée de Leaky ReLU."""
    return C.leaky_relu_deriv(x, alpha)


# ==================================================================
# Réseau générateur (nombre de couches variable)
# ==================================================================

def _generator(z, G_params, get_hiddens=False, apply_tanh=True):
    """
    Passe avant du générateur.

    Architecture variable : les paramètres utilisent des clés indexées
    G_W_0, G_b_0, G_W_1, G_b_1, …, G_W_{N-1}, G_b_{N-1} où N est
    le nombre de couches de poids.

    Activation : leaky_relu entre les couches, tanh (optionnel) sur
    la dernière couche.

    Args:
        z: Bruit latent (batch_size, latent_dim).
        G_params: Dictionnaire des paramètres avec clés indexées.
        get_hiddens: Si True, retourne les valeurs intermédiaires.
        apply_tanh: Si True, applique tanh sur la sortie.

    Returns:
        Si get_hiddens=False: output (batch_size, output_dim)
        Si get_hiddens=True: (output, z_list, h_list)
            z_list: [z_0, z_1, …, z_{N-1}] (pré-activations)
            h_list: [h_0, h_1, …, h_{N-2}] (post-activations des
                    couches cachées)
    """
    # Déterminer le nombre de couches de poids
    n_layers = 0
    while f'G_W_{n_layers}' in G_params:
        n_layers += 1

    h_prev = z
    z_list = []
    h_list = []

    for i in range(n_layers):
        W = G_params[f'G_W_{i}']
        b = G_params[f'G_b_{i}']
        z_i = C.matmul(h_prev, W) + b
        z_list.append(z_i)

        if i < n_layers - 1:
            h_i = leaky_relu(z_i)
            h_list.append(h_i)
            h_prev = h_i

    # Activation de la dernière couche
    if apply_tanh:
        output = C.tanh(z_list[-1])
    else:
        output = z_list[-1]

    if get_hiddens:
        return output, z_list, h_list
    return output


# ==================================================================
# Réseau discriminateur (nombre de couches variable)
# ==================================================================

def _discriminator(x, D_params, get_hiddens=False):
    """
    Passe avant du discriminateur.

    Architecture variable : les paramètres utilisent des clés indexées
    D_W_0, D_b_0, D_W_1, D_b_1, …, D_W_{N-1}, D_b_{N-1}.

    Activation : leaky_relu entre les couches, sigmoïde sur la
    dernière.

    Args:
        x: Données d'entrée (batch_size, data_dim).
        D_params: Dictionnaire des paramètres avec clés indexées.
        get_hiddens: Si True, retourne les valeurs intermédiaires.

    Returns:
        Si get_hiddens=False: output (batch_size, 1)
        Si get_hiddens=True: (output, z_list, h_list)
            z_list: [z_0, z_1, …, z_{N-1}] (pré-activations)
            h_list: [h_0, h_1, …, h_{N-2}] (post-activations des
                    couches cachées)
    """
    # Déterminer le nombre de couches de poids
    n_layers = 0
    while f'D_W_{n_layers}' in D_params:
        n_layers += 1

    h_prev = x
    z_list = []
    h_list = []

    for i in range(n_layers):
        W = D_params[f'D_W_{i}']
        b = D_params[f'D_b_{i}']
        z_i = C.matmul(h_prev, W) + b
        z_list.append(z_i)

        if i < n_layers - 1:
            h_i = leaky_relu(z_i)
            h_list.append(h_i)
            h_prev = h_i

    output = sigmoid(z_list[-1])

    if get_hiddens:
        return output, z_list, h_list
    return output


# ==================================================================
# Boucle d'entraînement GAN partagée
# ==================================================================

def _train_gan_core(data_dim, latent_dim, hidden_dim, lr, epochs, batch_size,
                    save_path, seed, filename, apply_tanh,
                    real_data_fn, extra_save_keys,
                    generator_layers=None, discriminator_layers=None):
    """
    Boucle d'entraînement GAN commune à toutes les variantes.

    Le discriminateur est entraîné 2 fois par étape de générateur.
    Objectif : log-loss GAN standard (non-saturating pour G).

    Args:
        data_dim: Dimension des données de sortie.
        latent_dim: Dimension de l'espace latent.
        hidden_dim: Dimension cachée (utilisé uniquement si les
            *layers correspondants sont None).
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        batch_size: Taille de batch.
        save_path: Chemin de sauvegarde (None = défaut).
        seed: Graine aléatoire.
        filename: Nom de fichier pour la sauvegarde.
        apply_tanh: Appliquer tanh sur la sortie du générateur.
        real_data_fn: Fonction(batch_size) → données réelles.
        extra_save_keys: Clés supplémentaires à sauvegarder.
        generator_layers: Liste des tailles de couches du générateur,
            p.ex. [latent_dim, 128, 64, data_dim]. Dernier élément =
            dimension de sortie. Si None, utilise l'architecture par
            défaut à 3 couches.
        discriminator_layers: Liste des tailles de couches du
            discriminateur, p.ex. [data_dim, 128, 64, 1]. Dernier
            élément = 1. Si None, utilise l'architecture par défaut
            à 3 couches.
    """
    global _seed_counter
    _seed_counter = seed
    np.random.seed(seed)
    random.seed(seed)

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"{filename}{MODEL_EXTENSION}")

    # --- Résoudre les architectures de couches ---
    gen_was_none = generator_layers is None
    disc_was_none = discriminator_layers is None

    if generator_layers is None:
        generator_layers = [latent_dim, hidden_dim, hidden_dim, data_dim]
    if discriminator_layers is None:
        discriminator_layers = [data_dim, hidden_dim, hidden_dim, 1]

    n_G = len(generator_layers) - 1   # nombre de couches de poids G
    n_D = len(discriminator_layers) - 1  # nombre de couches de poids D

    # --- Initialisation des poids ---
    G_params = {}
    for i in range(n_G):
        fan_in = generator_layers[i]
        fan_out = generator_layers[i + 1]
        G_params[f'G_W_{i}'] = xavier_init((fan_in, fan_out))
        G_params[f'G_b_{i}'] = C.zeros((fan_out,))

    D_params = {}
    for i in range(n_D):
        fan_in = discriminator_layers[i]
        fan_out = discriminator_layers[i + 1]
        D_params[f'D_W_{i}'] = xavier_init((fan_in, fan_out))
        D_params[f'D_b_{i}'] = C.zeros((fan_out,))

    history_D = []
    history_G = []

    log_interval = max(1, epochs // 10)

    for epoch in range(epochs):
        # ---- Entraîner le discriminateur 2 fois ----
        for _ in range(2):
            real_data = real_data_fn(batch_size)

            # Forward D sur données réelles
            D_real, z_list_r, h_list_r = _discriminator(
                real_data, D_params, get_hiddens=True
            )

            # Forward G pour générer des fausses données
            z = C.randn((batch_size, latent_dim), _next_seed())
            fake_data = _generator(
                z, G_params, get_hiddens=False, apply_tanh=apply_tanh
            )

            # Forward D sur fausses données
            D_fake, z_list_f, h_list_f = _discriminator(
                fake_data, D_params, get_hiddens=True
            )

            D_loss = (-C.mean(C.log(C.add_scalar(D_real, 1e-8)))
                      - C.mean(C.log(C.add_scalar(
                          C.sub(C.ones(D_fake.shape), D_fake), 1e-8))))

            # -- Rétropropagation D : chemin réel --
            # Gradient de D_loss w.r.t. D_real (sortie sigmoïde)
            d_D_real = C.neg(C.div(
                C.ones(D_real.shape), C.add_scalar(D_real, 1e-8)))
            # Dérivée sigmoïde sur la dernière couche
            d_z_r = C.mul(d_D_real, C.mul(
                D_real, C.sub(C.ones(D_real.shape), D_real)))

            # Accumulateurs de gradients D
            d_DW = [None] * n_D
            d_Db = [None] * n_D

            for i in range(n_D - 1, -1, -1):
                if i == 0:
                    h_in = real_data
                else:
                    h_in = h_list_r[i - 1]
                d_DW[i] = C.matmul(h_in.T, d_z_r)
                d_Db[i] = C.sum_axis(d_z_r, 0)
                if i > 0:
                    d_h = C.matmul(d_z_r, D_params[f'D_W_{i}'].T)
                    d_z_r = C.mul(d_h, leaky_relu_deriv(z_list_r[i - 1]))

            # -- Rétropropagation D : chemin fake --
            d_D_fake = C.div(
                C.ones(D_fake.shape),
                C.add_scalar(C.sub(C.ones(D_fake.shape), D_fake), 1e-8))
            d_z_f = C.mul(d_D_fake, C.mul(
                D_fake, C.sub(C.ones(D_fake.shape), D_fake)))

            for i in range(n_D - 1, -1, -1):
                if i == 0:
                    h_in = fake_data
                else:
                    h_in = h_list_f[i - 1]
                d_DW[i] = d_DW[i] + C.matmul(h_in.T, d_z_f)
                d_Db[i] = d_Db[i] + C.sum_axis(d_z_f, 0)
                if i > 0:
                    d_h = C.matmul(d_z_f, D_params[f'D_W_{i}'].T)
                    d_z_f = C.mul(d_h, leaky_relu_deriv(z_list_f[i - 1]))

            # -- Mise à jour D --
            for i in range(n_D - 1, -1, -1):
                D_params[f'D_W_{i}'] = C.sub(
                    D_params[f'D_W_{i}'],
                    C.scale(d_DW[i], lr / batch_size))
                D_params[f'D_b_{i}'] = C.sub(
                    D_params[f'D_b_{i}'],
                    C.scale(d_Db[i], lr / batch_size))

        # ---- Entraîner le générateur ----
        z = C.randn((batch_size, latent_dim), _next_seed())
        fake_data, g_z_list, g_h_list = _generator(
            z, G_params, get_hiddens=True, apply_tanh=apply_tanh
        )
        D_fake, d_z_list_d, d_h_list_d = _discriminator(
            fake_data, D_params, get_hiddens=True
        )

        G_loss = -C.mean(C.log(C.add_scalar(D_fake, 1e-8)))

        # -- Rétropropagation G à travers D (sans gradients pour D) --
        d_D_fake = C.neg(C.div(
            C.ones(D_fake.shape), C.add_scalar(D_fake, 1e-8)))
        d_z_d = C.mul(d_D_fake, C.mul(
            D_fake, C.sub(C.ones(D_fake.shape), D_fake)))

        # Propager à travers les couches cachées de D, puis obtenir
        # le gradient w.r.t. l'entrée de D (= sortie de G)
        for i in range(n_D - 1, 0, -1):
            d_h = C.matmul(d_z_d, D_params[f'D_W_{i}'].T)
            d_z_d = C.mul(d_h, leaky_relu_deriv(d_z_list_d[i - 1]))
        d_fake_data = C.matmul(d_z_d, D_params[f'D_W_0'].T)

        # -- Rétropropagation à travers G --
        if apply_tanh:
            d_z_g = C.mul(d_fake_data,
                           C.sub(C.ones(fake_data.shape),
                                 C.pow(fake_data, 2)))
        else:
            d_z_g = d_fake_data

        d_GW = [None] * n_G
        d_Gb = [None] * n_G

        for i in range(n_G - 1, -1, -1):
            if i == 0:
                h_in = z
            else:
                h_in = g_h_list[i - 1]
            d_GW[i] = C.matmul(h_in.T, d_z_g)
            d_Gb[i] = C.sum_axis(d_z_g, 0)
            if i > 0:
                d_h = C.matmul(d_z_g, G_params[f'G_W_{i}'].T)
                d_z_g = C.mul(d_h, leaky_relu_deriv(g_z_list[i - 1]))

        # -- Mise à jour G --
        for i in range(n_G - 1, -1, -1):
            G_params[f'G_W_{i}'] = C.sub(
                G_params[f'G_W_{i}'],
                C.scale(d_GW[i], lr / batch_size))
            G_params[f'G_b_{i}'] = C.sub(
                G_params[f'G_b_{i}'],
                C.scale(d_Gb[i], lr / batch_size))

        history_D.append(float(D_loss))
        history_G.append(float(G_loss))

        if epoch % log_interval == 0:
            logger.info(
                "Epoch %d/%d  D_loss: %.6f  G_loss: %.6f",
                epoch, epochs, D_loss, G_loss,
            )

    # --- Évaluation : précision du discriminateur ---
    n_eval = 200
    real_eval = real_data_fn(n_eval)
    z_eval = C.randn((n_eval, latent_dim), _next_seed())
    fake_eval = _generator(z_eval, G_params, apply_tanh=apply_tanh)
    D_real_eval = _discriminator(real_eval, D_params)
    D_fake_eval = _discriminator(fake_eval, D_params)
    accuracy = float(
        (np.mean(D_real_eval > 0.5) + np.mean(D_fake_eval < 0.5)) / 2.0
    )

    # --- Construction du modèle sauvegardé (générateur seul) ---
    model = {}
    for i in range(n_G):
        model[f'G_W_{i}'] = G_params[f'G_W_{i}']
        model[f'G_b_{i}'] = G_params[f'G_b_{i}']

    # Clés de rétrocompatibilité quand l'architecture par défaut
    # (3 couches) est utilisée
    if gen_was_none:
        for i in range(n_G):
            model[f'G_W{i + 1}'] = G_params[f'G_W_{i}']
            model[f'G_b{i + 1}'] = G_params[f'G_b_{i}']

    model['latent_dim'] = latent_dim
    model['hidden_dim'] = hidden_dim

    # Sauvegarder la configuration des couches si non par défaut
    if not gen_was_none:
        model['generator_layers'] = generator_layers
    if not disc_was_none:
        model['discriminator_layers'] = discriminator_layers

    model.update(extra_save_keys)

    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

    logger.info(
        "%s sauvegardé dans %s (accuracy D: %.4f)", filename, save_path, accuracy
    )

    history = {'D_loss': history_D, 'G_loss': history_G}

    return {
        'model': model,
        'save_path': save_path,
        'accuracy': accuracy,
        'history': history,
    }


# ==================================================================
# GAN 1D
# ==================================================================

def _real_data_1d(batch_size):
    """Distribution réelle : mélange de 2 gaussiennes N(-2, 0.5) et N(+2, 0.5)."""
    mask = (np.random.rand(batch_size, 1) > 0.5).astype(float)
    samples = (mask * (np.random.randn(batch_size, 1) * 0.5 + 2.0)
               + (1.0 - mask) * (np.random.randn(batch_size, 1) * 0.5 - 2.0))
    return samples


def train_gan_1d(latent_dim=2, hidden_dim=16, lr=0.001, epochs=5000,
                 batch_size=32, save_path=None, seed=42,
                 generator_layers=None, discriminator_layers=None):
    """
    Entraîne un GAN 1D générant une valeur scalaire.

    Données réelles : mélange de 2 gaussiennes (moyennes -2 et +2,
    écart-type 0.5). Pas de tanh sur la sortie du générateur.

    Args:
        latent_dim: Dimension de l'espace latent.
        hidden_dim: Dimension cachée (si *layers est None).
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        batch_size: Taille de batch.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        generator_layers: Liste des tailles de couches du générateur.
            Si None, [latent_dim, hidden_dim, hidden_dim, 1].
        discriminator_layers: Liste des tailles de couches du
            discriminateur. Si None, [1, hidden_dim, hidden_dim, 1].

    Returns:
        dict: {'model', 'save_path', 'accuracy', 'history'}
    """
    return _train_gan_core(
        data_dim=1,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
        seed=seed,
        filename="gan_1d",
        apply_tanh=False,
        real_data_fn=_real_data_1d,
        extra_save_keys={},
        generator_layers=generator_layers,
        discriminator_layers=discriminator_layers,
    )


# ==================================================================
# GAN N-D
# ==================================================================

def _real_data_nd(batch_size, data_dim):
    """Distribution réelle N-D : chaque dimension échantillonnée
    indépendamment de N(-2, 0.5) ou N(+2, 0.5)."""
    mask = (np.random.rand(batch_size, data_dim) > 0.5).astype(float)
    samples = (mask * (np.random.randn(batch_size, data_dim) * 0.5 + 2.0)
               + (1.0 - mask) * (np.random.randn(batch_size, data_dim) * 0.5 - 2.0))
    return samples


def train_gan_nd(data_dim=16, latent_dim=16, hidden_dim=64, lr=0.001,
                 epochs=5000, batch_size=32, save_path=None, seed=42,
                 generator_layers=None, discriminator_layers=None):
    """
    Entraîne un GAN N-D générant des vecteurs de dimension data_dim.

    Données réelles : mélange de gaussiennes par dimension.
    Tanh sur la sortie du générateur.

    Args:
        data_dim: Dimension des données.
        latent_dim: Dimension de l'espace latent.
        hidden_dim: Dimension cachée (si *layers est None).
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        batch_size: Taille de batch.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        generator_layers: Liste des tailles de couches du générateur.
            Si None, [latent_dim, hidden_dim, hidden_dim, data_dim].
        discriminator_layers: Liste des tailles de couches du
            discriminateur. Si None, [data_dim, hidden_dim, hidden_dim, 1].

    Returns:
        dict: {'model', 'save_path', 'accuracy', 'history'}
    """
    fn = lambda bs: _real_data_nd(bs, data_dim)

    return _train_gan_core(
        data_dim=data_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
        seed=seed,
        filename="gan_nd",
        apply_tanh=True,
        real_data_fn=fn,
        extra_save_keys={'data_dim': data_dim},
        generator_layers=generator_layers,
        discriminator_layers=discriminator_layers,
    )


# ==================================================================
# GAN 3D
# ==================================================================

def _real_data_3d(batch_size, volume_size):
    """Distribution réelle 3D : sphères ou cubes avec bruit."""
    data_dim = volume_size ** 3
    gx, gy, gz = C.mgrid3d(volume_size)
    coords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    center = (volume_size - 1) / 2.0
    distances = np.sqrt(np.sum((coords - center) ** 2, axis=1))

    batches = np.zeros((batch_size, data_dim))
    for i in range(batch_size):
        shape_type = random.randint(0, 1)
        noise = np.random.randn(data_dim) * 0.1
        if shape_type == 0:
            # Sphère
            radius = random.uniform(0.8, 1.8)
            volumes = (distances <= radius).astype(float)
        else:
            # Cube
            half_size = random.uniform(0.5, 1.5)
            inside = np.all(np.abs(coords - center) <= half_size, axis=1)
            volumes = inside.astype(float)
        batches[i] = volumes + noise

    # Normaliser vers [-1, 1]
    batches = np.clip(batches * 2.0 - 1.0, -1.0, 1.0)
    return batches


def train_gan_3d(volume_size=4, latent_dim=16, hidden_dim=128, lr=0.0005,
                 epochs=5000, batch_size=32, save_path=None, seed=42,
                 generator_layers=None, discriminator_layers=None):
    """
    Entraîne un GAN 3D générant des volumes 4x4x4 (64 voxels).

    Données réelles : sphères ou cubes avec bruit.
    Tanh sur la sortie du générateur.

    Args:
        volume_size: Taille du volume (volume_size^3 voxels).
        latent_dim: Dimension de l'espace latent.
        hidden_dim: Dimension cachée (si *layers est None).
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        batch_size: Taille de batch.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        generator_layers: Liste des tailles de couches du générateur.
            Si None, [latent_dim, hidden_dim, hidden_dim, data_dim].
        discriminator_layers: Liste des tailles de couches du
            discriminateur. Si None, [data_dim, hidden_dim, hidden_dim, 1].

    Returns:
        dict: {'model', 'save_path', 'accuracy', 'history'}
    """
    data_dim = volume_size ** 3
    fn = lambda bs: _real_data_3d(bs, volume_size)

    extra = {
        'VOLUME_H': volume_size,
        'VOLUME_W': volume_size,
        'VOLUME_D': volume_size,
        'volume_shape': (volume_size, volume_size, volume_size),
    }

    return _train_gan_core(
        data_dim=data_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
        seed=seed,
        filename="gan_3d",
        apply_tanh=True,
        real_data_fn=fn,
        extra_save_keys=extra,
        generator_layers=generator_layers,
        discriminator_layers=discriminator_layers,
    )


# ==================================================================
# GAN RGB
# ==================================================================

def _real_data_rgb(batch_size, image_size, channels):
    """Distribution réelle RGB : motifs de couleur synthétiques."""
    flat_dim = image_size * image_size * channels
    batches = np.zeros((batch_size, flat_dim))

    for i in range(batch_size):
        pattern = random.randint(0, 4)
        img = np.zeros((image_size, image_size, channels))

        if pattern == 0:
            # Dégradé horizontal
            for c in range(channels):
                start = random.uniform(-0.8, 0.8)
                end = random.uniform(-0.8, 0.8)
                ls = C.linspace(start, end, image_size)
                img[:, :, c] = ls.reshape(-1, 1)

        elif pattern == 1:
            # Formes géométriques colorées
            img = np.random.uniform(-0.3, 0.3, (image_size, image_size, channels))
            margin = max(1, image_size // 8)
            cx = random.randint(margin, max(margin + 1, image_size - margin) - 1)
            cy = random.randint(margin, max(margin + 1, image_size - margin) - 1)
            r = random.randint(1, max(2, image_size // 2) - 1)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy * dy + dx * dx <= r * r:
                        py, px = cy + dy, cx + dx
                        if 0 <= py < image_size and 0 <= px < image_size:
                            img[py, px] = np.random.uniform(0.5, 1.0, channels)

        elif pattern == 2:
            # Rayures verticales
            width = random.randint(2, 4)
            color_a = np.random.uniform(-0.8, 0.8, channels)
            color_b = np.random.uniform(-0.8, 0.8, channels)
            for j in range(image_size):
                c = color_a if (j // width) % 2 == 0 else color_b
                img[:, j, :] = c

        elif pattern == 3:
            # Bruit coloré
            img = np.random.uniform(-1.0, 1.0, (image_size, image_size, channels))

        elif pattern == 4:
            # Motif mixte : deux moitiés avec couleurs différentes
            half = image_size // 2
            img[:half, :, :] = np.random.uniform(0.3, 1.0, (half, image_size, channels))
            img[half:, :, :] = np.random.uniform(-1.0, -0.3, (image_size - half, image_size, channels))

        batches[i] = img.flatten()

    batches = np.clip(batches, -1.0, 1.0)
    return batches


def train_gan_rgb(image_size=32, channels=3, latent_dim=200, hidden_dim=512,
                  lr=0.00002, epochs=10, batch_size=16, save_path=None, seed=42,
                  generator_layers=None, discriminator_layers=None):
    """
    Entraîne un GAN RGB générant des images 32x32x3 (3072 pixels).

    Données réelles : motifs de couleur synthétiques (dégradés, formes,
    rayures, bruit, mixte). Tanh sur la sortie du générateur.

    Args:
        image_size: Taille des images (carrées).
        channels: Nombre de canaux de couleur.
        latent_dim: Dimension de l'espace latent.
        hidden_dim: Dimension cachée (si *layers est None).
        lr: Taux d'apprentissage.
        epochs: Nombre d'époques.
        batch_size: Taille de batch.
        save_path: Chemin de sauvegarde.
        seed: Graine aléatoire.
        generator_layers: Liste des tailles de couches du générateur.
            Si None, [latent_dim, hidden_dim, hidden_dim, data_dim].
        discriminator_layers: Liste des tailles de couches du
            discriminateur. Si None, [data_dim, hidden_dim, hidden_dim, 1].

    Returns:
        dict: {'model', 'save_path', 'accuracy', 'history'}
    """
    data_dim = image_size * image_size * channels
    fn = lambda bs: _real_data_rgb(bs, image_size, channels)

    extra = {
        'IMAGE_H': image_size,
        'IMAGE_W': image_size,
        'CHANNELS': channels,
        'image_shape': (image_size, image_size, channels),
    }

    return _train_gan_core(
        data_dim=data_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
        seed=seed,
        filename="gan_rgb",
        apply_tanh=True,
        real_data_fn=fn,
        extra_save_keys=extra,
        generator_layers=generator_layers,
        discriminator_layers=discriminator_layers,
    )