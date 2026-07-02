"""
IA/infer/gan.py — Inférence des GAN (génération de données).

Fonctions :
  - load_gan_1d / generate_gan_1d       : GAN 1D
  - load_gan_nd / generate_gan_nd       : GAN N-D
  - load_gan_3d / generate_gan_3d       : GAN 3D
  - load_gan_rgb / generate_gan_rgb     : GAN RGB
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from .._utils import merge_header, validate_model_path
from ..cpp import get_core
from ..ia_format import load_model

C = get_core()

logger = logging.getLogger(__name__)


# ==================================================================
# Helpers
# ==================================================================

def leaky_relu(x, alpha=0.01):
    return C.leaky_relu(x, alpha)

def tanh(x):
    return C.tanh(x)

def sigmoid(x):
    return C.sigmoid(x)


def _load_gan(path: str) -> Dict[str, Any]:
    """Charge un modèle GAN depuis un fichier .gy (format V2/V3).

    Returns:
        dict: Paramètres du générateur prêts pour generate_gan_*.
    """
    path = validate_model_path(path)
    header, tensors = load_model(path)
    model = merge_header(header, tensors)

    # Convertir les listes JSON en tuples pour volume_shape, image_shape
    for key in ('volume_shape', 'image_shape'):
        if key in model and isinstance(model[key], list):
            model[key] = tuple(model[key])

    logger.info("GAN chargé depuis %s", path)
    return model


def _generator_forward(z: np.ndarray, params: Dict[str, Any],
                        output_activation=None) -> np.ndarray:
    """Passe avant du générateur GAN (format indexé G_W_0, G_b_0, …)."""
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
    if output_activation is not None:
        out = output_activation(out)
    return out


def _get_gan_latent_dim(model: Dict[str, Any]) -> int:
    return model['G_W_0'].shape[0]


def _get_gan_output_dim(model: Dict[str, Any]) -> int:
    i = 0
    while f'G_W_{i}' in model:
        i += 1
    return model[f'G_W_{i - 1}'].shape[1]


# ==================================================================
# GAN 1D
# ==================================================================

def load_gan_1d(path: str) -> Dict[str, Any]:
    """Charge les paramètres du générateur GAN 1D depuis un fichier .gy."""
    return _load_gan(path)


def generate_gan_1d(model: Dict[str, Any], z=None, num_samples=1) -> Dict[str, Any]:
    """Génère des données 1D à partir du générateur GAN 1D."""
    latent_dim = _get_gan_latent_dim(model)
    if z is None:
        z = np.random.randn(num_samples, latent_dim)
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    generated = _generator_forward(z, model, output_activation=None)
    logger.info("GAN 1D — %d échantillon(s) généré(s), forme: %s", generated.shape[0], generated.shape)
    return {'generated': generated, 'latent_vectors': z}


# ==================================================================
# GAN N-D
# ==================================================================

def load_gan_nd(path: str) -> Dict[str, Any]:
    """Charge les paramètres du générateur GAN N-D depuis un fichier .gy."""
    return _load_gan(path)


def generate_gan_nd(model: Dict[str, Any], z=None, num_samples=1) -> Dict[str, Any]:
    """Génère des vecteurs N-D à partir du générateur GAN N-D."""
    latent_dim = _get_gan_latent_dim(model)
    if z is None:
        z = np.random.randn(num_samples, latent_dim)
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    generated = _generator_forward(z, model, output_activation=tanh)
    logger.info("GAN N-D — %d échantillon(s) généré(s), forme: %s", generated.shape[0], generated.shape)
    return {'generated': generated, 'latent_vectors': z}


# ==================================================================
# GAN 3D
# ==================================================================

def load_gan_3d(path: str) -> Dict[str, Any]:
    """Charge les paramètres du générateur GAN 3D depuis un fichier .gy."""
    return _load_gan(path)


def generate_gan_3d(model: Dict[str, Any], z=None) -> Dict[str, Any]:
    """Génère un volume 3D à partir du générateur GAN 3D."""
    latent_dim = _get_gan_latent_dim(model)
    volume_shape = model['volume_shape']
    if z is None:
        z = np.random.randn(1, latent_dim)
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    flat = _generator_forward(z, model, output_activation=tanh)
    generated_volume = flat[0].reshape(volume_shape)
    logger.info("GAN 3D — volume généré, forme: %s", volume_shape)
    return {'generated_volume': generated_volume, 'volume_shape': volume_shape}


# ==================================================================
# GAN RGB
# ==================================================================

def load_gan_rgb(path: str) -> Dict[str, Any]:
    """Charge les paramètres du générateur GAN RGB depuis un fichier .gy."""
    return _load_gan(path)


def generate_gan_rgb(model: Dict[str, Any], z=None, num_samples=1) -> Dict[str, Any]:
    """Génère des images RGB à partir du générateur GAN RGB."""
    latent_dim = _get_gan_latent_dim(model)
    image_shape = model['image_shape']
    if z is None:
        z = np.random.randn(num_samples, latent_dim)
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    flat = _generator_forward(z, model, output_activation=tanh)
    h, w = image_shape
    generated_images = flat.reshape(num_samples, h, w, 3)
    logger.info("GAN RGB — %d image(s) générée(s), forme: %s", num_samples, image_shape)
    return {'generated_images': generated_images, 'image_shape': image_shape}