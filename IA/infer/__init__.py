"""
IA/infer — Sous-paquet inférence.

Fournit les fonctions de chargement et de prédiction pour tous les modèles.
"""

from .cnn import load_cnn2d, predict_cnn2d, load_cnn_nd, predict_cnn_nd
from .rnn import load_rnn, predict_rnn
from .transformer import load_transformer, predict_transformer, load_transformer3d, predict_transformer3d
from .gan import (load_gan_1d, generate_gan_1d, load_gan_nd, generate_gan_nd,
                  load_gan_3d, generate_gan_3d, load_gan_rgb, generate_gan_rgb)
from .ldm import load_ldm_image, generate_ldm_image, load_ldm_audio, generate_ldm_audio
from .slm import load_slm, predict_slm, predict_slm_next_word
from .vision import load_image_classifier, predict_image, load_speech_classifier, predict_speech
from .mlp import load_mlp, predict_mlp

__all__ = [
    "load_cnn2d", "predict_cnn2d", "load_cnn_nd", "predict_cnn_nd",
    "load_rnn", "predict_rnn",
    "load_transformer", "predict_transformer", "load_transformer3d", "predict_transformer3d",
    "load_gan_1d", "generate_gan_1d", "load_gan_nd", "generate_gan_nd",
    "load_gan_3d", "generate_gan_3d", "load_gan_rgb", "generate_gan_rgb",
    "load_ldm_image", "generate_ldm_image", "load_ldm_audio", "generate_ldm_audio",
    "load_slm", "predict_slm", "predict_slm_next_word",
    "load_image_classifier", "predict_image", "load_speech_classifier", "predict_speech",
    "load_mlp", "predict_mlp",
]