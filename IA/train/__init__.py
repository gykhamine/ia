"""
IA/train — Sous-paquet entraînement.

Fournit les fonctions d'entraînement pour tous les modèles : CNN, RNN, Transformer, GAN, LDM, SLM, Vision.
"""

from .cnn import train_cnn2d, train_cnn_nd
from .rnn import train_rnn
from .transformer import train_transformer, train_transformer3d, MiniTransformer3D
from .gan import train_gan_1d, train_gan_nd, train_gan_3d, train_gan_rgb
from .ldm import train_ldm_image, train_ldm_audio
from .slm import (train_slm_next_word, train_slm_emotion, train_slm_mood,
                  train_slm_statement, train_slm_sentiment, SLMClassifier, TextPreprocessor)
from .mlp import train_mlp
from .vision import train_image_classifier, train_speech_classifier

__all__ = [
    "train_mlp",
    "train_cnn2d", "train_cnn_nd",
    "train_rnn",
    "train_transformer", "train_transformer3d", "MiniTransformer3D",
    "train_gan_1d", "train_gan_nd", "train_gan_3d", "train_gan_rgb",
    "train_ldm_image", "train_ldm_audio",
    "train_slm_next_word", "train_slm_emotion", "train_slm_mood",
    "train_slm_statement", "train_slm_sentiment", "SLMClassifier", "TextPreprocessor",
    "train_image_classifier", "train_speech_classifier",
]
