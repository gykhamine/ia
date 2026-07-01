"""
IA — Module d'Intelligence Artificielle avec moteur de calcul C++.

Architecture modulaire pour l'entrainement et l'inference de modeles deep learning
avec un moteur de calcul C++ pur (ctypes) remplacant numpy/pandas pour
toutes les operations mathematiques fondamentales.

Organisation :
  - cpp/        : Moteur de calcul C++ (engine.h, engine.cpp, c_api.cpp, autograd.cpp)
  - train/      : Entrainement de tous les modeles (CNN, RNN, Transformer, GAN, LDM, SLM, Vision).
  - infer/      : Chargement et inference de tous les modeles entraines.
  - dataset.py  : Chargement multi-formats (CSV, JSON, Parquet, HDF5, TXT, etc.)
  - trainer.py  : Classe Trainer unifiee (V3) + Model + callbacks
  - model.py    : Classe Model (wrapper d'inference + persistance .ia)
  - callbacks.py: Callbacks Keras-like (EarlyStopping, ModelCheckpoint, ...)
  - ia_format.py: Format binaire natif .ia pour la persistance des modeles

API unifiee (V3) :
    from IA import Trainer
    trainer = Trainer()
    model = trainer.train(type='rnn', epochs=500)
    y = model.predict(X)
    model.save('mon_modele.ia')
    model2 = Trainer.load('mon_modele.ia')

API bas niveau (rétro-compatible V2) :
    from IA import train_rnn
    result = train_rnn(epochs=500)
"""

import logging

_logging_configured = False


def _setup_logging():
    global _logging_configured
    if _logging_configured:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _logging_configured = True


_setup_logging()

# --- Config ---
from .config import get_config, ensure_directories, MODELS_DIR, MODEL_EXTENSION

# --- Moteur C++ ---
from . import cpp

# --- Sous-modules ---
from .dataset import load_dataset, supported_formats, dataset_info

from . import train, infer

# --- V3 : Trainer, Model, callbacks, format .ia ---
from .trainer import Trainer
from .model import Model
from . import callbacks
from .callbacks import (
    Callback, CallbackList,
    EarlyStopping, ModelCheckpoint, ProgressPrinter, CSVLogger,
)
from . import ia_format

# --- Raccourcis entraînement (rétro-compatibilité V2) ---
train_cnn2d = train.train_cnn2d
train_cnn_nd = train.train_cnn_nd
train_rnn = train.train_rnn
train_transformer = train.train_transformer
train_transformer3d = train.train_transformer3d
train_gan_1d = train.train_gan_1d
train_gan_nd = train.train_gan_nd
train_gan_3d = train.train_gan_3d
train_gan_rgb = train.train_gan_rgb
train_ldm_image = train.train_ldm_image
train_ldm_audio = train.train_ldm_audio
train_slm_next_word = train.train_slm_next_word
train_slm_emotion = train.train_slm_emotion
train_slm_mood = train.train_slm_mood
train_slm_statement = train.train_slm_statement
train_slm_sentiment = train.train_slm_sentiment
train_image_classifier = train.train_image_classifier
train_speech_classifier = train.train_speech_classifier
train_mlp = train.train_mlp

# --- Raccourcis inférence ---
load_cnn2d = infer.load_cnn2d
predict_cnn2d = infer.predict_cnn2d
load_cnn_nd = infer.load_cnn_nd
predict_cnn_nd = infer.predict_cnn_nd
load_rnn = infer.load_rnn
predict_rnn = infer.predict_rnn
load_transformer = infer.load_transformer
predict_transformer = infer.predict_transformer
load_transformer3d = infer.load_transformer3d
predict_transformer3d = infer.predict_transformer3d
load_gan_1d = infer.load_gan_1d
generate_gan_1d = infer.generate_gan_1d
load_gan_nd = infer.load_gan_nd
generate_gan_nd = infer.generate_gan_nd
load_gan_3d = infer.load_gan_3d
generate_gan_3d = infer.generate_gan_3d
load_gan_rgb = infer.load_gan_rgb
generate_gan_rgb = infer.generate_gan_rgb
load_ldm_image = infer.load_ldm_image
generate_ldm_image = infer.generate_ldm_image
load_ldm_audio = infer.load_ldm_audio
generate_ldm_audio = infer.generate_ldm_audio
load_slm = infer.load_slm
predict_slm = infer.predict_slm
predict_slm_next_word = infer.predict_slm_next_word
load_image_classifier = infer.load_image_classifier
predict_image = infer.predict_image
load_speech_classifier = infer.load_speech_classifier
predict_speech = infer.predict_speech
load_mlp = infer.load_mlp
predict_mlp = infer.predict_mlp

__all__ = [
    # Config
    "get_config", "ensure_directories", "MODELS_DIR", "MODEL_EXTENSION",
    # V3 : API unifiée
    "Trainer", "Model",
    "callbacks", "ia_format",
    "Callback", "CallbackList",
    "EarlyStopping", "ModelCheckpoint", "ProgressPrinter", "CSVLogger",
    # Entraînement CNN
    "train_cnn2d", "train_cnn_nd",
    # Entraînement RNN
    "train_rnn",
    # Entraînement Transformer
    "train_transformer", "train_transformer3d",
    # Entraînement GAN
    "train_gan_1d", "train_gan_nd", "train_gan_3d", "train_gan_rgb",
    # Entraînement LDM
    "train_ldm_image", "train_ldm_audio",
    # Entraînement SLM
    "train_slm_next_word", "train_slm_emotion", "train_slm_mood",
    "train_slm_statement", "train_slm_sentiment",
    # Entraînement Vision
    "train_image_classifier", "train_speech_classifier",
    # Entraînement MLP
    "train_mlp",
    # Inférence CNN
    "load_cnn2d", "predict_cnn2d", "load_cnn_nd", "predict_cnn_nd",
    # Inférence RNN
    "load_rnn", "predict_rnn",
    # Inférence Transformer
    "load_transformer", "predict_transformer",
    "load_transformer3d", "predict_transformer3d",
    # Inférence GAN
    "load_gan_1d", "generate_gan_1d",
    "load_gan_nd", "generate_gan_nd",
    "load_gan_3d", "generate_gan_3d",
    "load_gan_rgb", "generate_gan_rgb",
    # Inférence LDM
    "load_ldm_image", "generate_ldm_image",
    "load_ldm_audio", "generate_ldm_audio",
    # Inférence SLM
    "load_slm", "predict_slm", "predict_slm_next_word",
    # Inférence Vision
    "load_image_classifier", "predict_image",
    "load_speech_classifier", "predict_speech",
    # Inférence MLP
    "load_mlp", "predict_mlp",
    # Sous-modules
    "load_dataset", "supported_formats", "dataset_info",
    "train", "infer",
]