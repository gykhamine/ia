"""
IA/infer/gan.py — Inférence des GAN (génération de données).
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

def leaky_relu(x, alpha=0.01):
    """Fonction d'activation Leaky ReLU."""
    return C.leaky_relu(x, alpha)


def tanh(x):
    """Fonction d'activation tanh."""
    return C.tanh(x)


def sigmoid(x):
    """Fonction d'activation sigmoïde."""
    return C.sigmoid(x)


def _is_new_gan_format(params):
    """Détecte le nouveau format indexé (G_W_0, G_b_0, …)."""
    return 'G_W_0' in params or 'generator_layers' in params


def _generator_forward(z, params, output_activation=None):
    """
    Passe avant générique du générateur GAN.

    Supporte deux formats :
      - Legacy : clés G_W1, G_b1, G_W2, G_b2, G_W3, G_b3
      - Nouveau : clés G_W_0, G_b_0, …, G_W_{N-1}, G_b_{N-1}

    Args:
        z: Vecteur latent (batch_size, latent_dim).
        params: Dictionnaire des paramètres du générateur.
        output_activation: Fonction d'activation pour la couche de sortie, ou None.

    Returns:
        ndarray: Données générées.
    """
    if _is_new_gan_format(params):
        # Format indexé : déterminer le nombre de couches
        i = 0
        while f'G_W_{i}' in params:
            i += 1
        n_layers = i
        h = z
        for layer_i in range(n_layers):
            W = params[f'G_W_{layer_i}']
            b = params[f'G_b_{layer_i}']
            z_l = C.matmul(h, W) + b
            if layer_i < n_layers - 1:
                h = leaky_relu(z_l)
            else:
                out = z_l
    else:
        # Format historique à 3 couches
        h1 = leaky_relu(C.matmul(z, params['G_W1']) + params['G_b1'])
        h2 = leaky_relu(C.matmul(h1, params['G_W2']) + params['G_b2'])
        out = C.matmul(h2, params['G_W3']) + params['G_b3']

    if output_activation is not None:
        out = output_activation(out)
    return out


def _get_gan_latent_dim(model):
    """Retourne la dimension latente du générateur."""
    if _is_new_gan_format(model):
        return model['G_W_0'].shape[0]
    return model.get('latent_dim', model['G_W1'].shape[0])


def _get_gan_output_dim(model):
    """Retourne la dimension de sortie du générateur."""
    if _is_new_gan_format(model):
        # Trouver la dernière couche
        i = 0
        while f'G_W_{i}' in model:
            i += 1
        return model[f'G_W_{i - 1}'].shape[1]
    return model['G_W3'].shape[1]


# ==================================================================
# GAN 1D
# ==================================================================

def load_gan_1d(path):
    """
    Charge les paramètres du générateur GAN 1D depuis un fichier pickle.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du générateur (G_W1, G_b1, G_W2, G_b2, G_W3, G_b3, etc.).
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("GAN 1D chargé depuis %s", path)
    return model


def generate_gan_1d(model, z=None, num_samples=1):
    """
    Génère des données 1D à partir du générateur GAN 1D.

    Args:
        model: Dictionnaire de paramètres (issu de load_gan_1d).
        z: Vecteur(s) latent(s) optionnel(s). Si None, tirés aléatoirement.
        num_samples: Nombre d'échantillons à générer (utilisé si z est None).

    Returns:
        dict: {
            'generated': ndarray(num_samples, output_dim),
            'latent_vectors': ndarray(num_samples, latent_dim),
        }
    """
    latent_dim = _get_gan_latent_dim(model)
    output_dim = _get_gan_output_dim(model)

    if z is None:
        z = np.random.randn(num_samples, latent_dim)
    z = np.asarray(z)
    if z.ndim == 1:
        z = z.reshape(1, -1)

    generated = _generator_forward(z, model, output_activation=None)

    logger.info("GAN 1D — %d échantillon(s) généré(s), forme: %s", generated.shape[0], generated.shape)

    return {
        'generated': generated,
        'latent_vectors': z,
    }


# ==================================================================
# GAN N-D
# ==================================================================

def load_gan_nd(path):
    """
    Charge les paramètres du générateur GAN N-D depuis un fichier pickle.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du générateur.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("GAN N-D chargé depuis %s", path)
    return model


def generate_gan_nd(model, z=None, num_samples=1):
    """
    Génère des vecteurs N-D à partir du générateur GAN N-D.

    Args:
        model: Dictionnaire de paramètres (issu de load_gan_nd).
        z: Vecteur(s) latent(s) optionnel(s). Si None, tirés aléatoirement.
        num_samples: Nombre d'échantillons à générer.

    Returns:
        dict: {
            'generated': ndarray(num_samples, output_dim),
            'latent_vectors': ndarray(num_samples, latent_dim),
        }
    """
    latent_dim = _get_gan_latent_dim(model)

    if z is None:
        z = np.random.randn(num_samples, latent_dim)
    z = np.asarray(z)
    if z.ndim == 1:
        z = z.reshape(1, -1)

    generated = _generator_forward(z, model, output_activation=tanh)

    logger.info("GAN N-D — %d échantillon(s) généré(s), forme: %s", generated.shape[0], generated.shape)

    return {
        'generated': generated,
        'latent_vectors': z,
    }


# ==================================================================
# GAN 3D
# ==================================================================

def load_gan_3d(path):
    """
    Charge les paramètres du générateur GAN 3D depuis un fichier pickle.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du générateur.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("GAN 3D chargé depuis %s", path)
    return model


def generate_gan_3d(model, z=None):
    """
    Génère un volume 3D à partir du générateur GAN 3D.

    Args:
        model: Dictionnaire de paramètres (issu de load_gan_3d).
        z: Vecteur latent optionnel. Si None, tiré aléatoirement.

    Returns:
        dict: {
            'generated_volume': ndarray(volume_shape),
            'volume_shape': tuple,
        }
    """
    latent_dim = _get_gan_latent_dim(model)
    volume_shape = model['volume_shape']

    if z is None:
        z = np.random.randn(1, latent_dim)
    z = np.asarray(z)
    if z.ndim == 1:
        z = z.reshape(1, -1)

    flat = _generator_forward(z, model, output_activation=tanh)
    generated_volume = flat[0].reshape(volume_shape)

    logger.info("GAN 3D — volume généré, forme: %s", volume_shape)

    return {
        'generated_volume': generated_volume,
        'volume_shape': volume_shape,
    }


# ==================================================================
# GAN RGB
# ==================================================================

def load_gan_rgb(path):
    """
    Charge les paramètres du générateur GAN RGB depuis un fichier pickle.

    Args:
        path: Chemin vers le fichier .gy sauvegardé.

    Returns:
        dict: Paramètres du générateur.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("GAN RGB chargé depuis %s", path)
    return model


def generate_gan_rgb(model, z=None, num_samples=1):
    """
    Génère des images RGB à partir du générateur GAN RGB.

    Args:
        model: Dictionnaire de paramètres (issu de load_gan_rgb).
        z: Vecteur(s) latent(s) optionnel(s). Si None, tirés aléatoirement.
        num_samples: Nombre d'images à générer.

    Returns:
        dict: {
            'generated_images': ndarray(num_samples, H, W, 3),
            'image_shape': tuple,
        }
    """
    latent_dim = _get_gan_latent_dim(model)
    image_shape = model['image_shape']

    if z is None:
        z = np.random.randn(num_samples, latent_dim)
    z = np.asarray(z)
    if z.ndim == 1:
        z = z.reshape(1, -1)

    flat = _generator_forward(z, model, output_activation=tanh)
    h, w = image_shape
    generated_images = flat.reshape(num_samples, h, w, 3)

    logger.info("GAN RGB — %d image(s) générée(s), forme: %s", num_samples, image_shape)

    return {
        'generated_images': generated_images,
        'image_shape': image_shape,
    }