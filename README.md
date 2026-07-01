<div align="center">
 <img src="https://gykhamine.github.io/GCI/statics/3.png" alt="Logo Gykhamine Studio" width="150" style="margin-bottom: 10px;" />
</div>

## IA V3 — Module d'Intelligence Artificielle

> Moteur de deep learning modulaire avec backend de calcul C++ pur (ctypes, sans pybind11 ni CMake), entraînement et inférence de 7 familles de modèles, format de persistance binaire natif `.ia`, callbacks Keras-like, et chargement de datasets multi-formats.

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Structure du projet](#structure-du-projet)
- [Prérequis et installation](#prérequis-et-installation)
  - [Dépendances système](#dépendances-système)
  - [Dépendances Python](#dépendances-python)
  - [Compilation du moteur C++](#compilation-du-moteur-c)
  - [Installation alternative (pip)](#installation-alternative-pip)
  - [Fallback numpy](#fallback-numpy)
- [Démarrage rapide](#démarrage-rapide)
  - [API V3 unifiée (recommandée)](#api-v3-unifiée-recommandée)
  - [API V2 rétro-compatible](#api-v2-rétro-compatible)
- [Modèles supportés](#modèles-supportés)
  - [RNN (Réseau Récurrent)](#rnn-réseau-récurrent)
  - [CNN (Réseau de Neurones Convolutif)](#cnn-réseau-de-neurones-convolutif)
  - [Transformer](#transformer)
  - [GAN (Réseau Génératif Adversarial)](#gan-réseau-génératif-adversarial)
  - [LDM (Modèle de Diffusion Latente)](#ldm-modèle-de-diffusion-latente)
  - [SLM (Small Language Model)](#slm-small-language-model)
  - [MLP (Perceptron Multi-Couches)](#mlp-perceptron-multi-couches)
  - [Vision (Classifieur d'images et de parole)](#vision-classifieur-dimages-et-de-parole)
- [Moteur de calcul C++](#moteur-de-calcul-c)
  - [Opérations disponibles](#opérations-disponibles)
  - [API zero-copy](#api-zero-copy)
  - [Autograd (Wengert Tape)](#autograd-wengert-tape)
- [Format binaire `.ia`](#format-binaire-ia)
- [Chargement de datasets](#chargement-de-datasets)
  - [Formats supportés](#formats-supportés)
  - [API](#api-1)
  - [Informations sur un dataset](#informations-sur-un-dataset)
- [Système de callbacks](#système-de-callbacks)
  - [Callback (base)](#callback-base)
  - [EarlyStopping](#earlystopping)
  - [ModelCheckpoint](#modelcheckpoint)
  - [ProgressPrinter](#progressprinter)
  - [CSVLogger](#csvlogger)
  - [Callbacks personnalisés](#callbacks-personnalisés)
- [Configuration](#configuration)
- [Modèles pré-entraînés inclus](#modèles-pré-entraînés-inclus)
- [Extension du framework](#extension-du-framework)
- [Limitations connues](#limitations-connues)
- [Changelog V2 → V3](#changelog-v2--v3)
- [ Licence](#licence)

---

## Vue d'ensemble

**IA V3** est un module Python d'intelligence artificielle qui implémente un moteur de calcul mathématique entièrement en **C++ pur** (standard C++17), sans aucune dépendance C++ externe — ni BLAS, ni LAPACK, ni Eigen, ni pybind11, ni CMake. L'interface Python est assurée exclusivement via **ctypes**, ce qui rend l'installation minimale : un simple `make` suffit pour compiler la shared library.

Le module couvre **7 familles de modèles deep learning** (CNN, RNN, Transformer, GAN, LDM, SLM, Vision), chacune avec une boucle d'entraînement complète (forward, loss, rétropropagation manuelle, mise à jour des poids) et un pipeline d'inférence dédié. Un format binaire natif `.ia` permet la persistance des modèles de manière portable et lisible depuis n'importe quel langage.

## Fonctionnalités

- **Moteur C++ pur** : Toutes les opérations mathématiques fondamentales (matmul, convolutions 1D/2D/ND, activations, normalisation, FFT, réductions) implémentées de zéro en C++17, compilées en shared library chargée via ctypes.
- **Autograd minimaliste** : Wengert tape (style micrograd/tinygrad) avec différenciation automatique pour les opérations de base (relu, sigmoid, tanh, add, sub, mul, scale, matmul, mse, dot).
- **8 architectures de modèles** : RNN vanilla (BPTT), CNN 2D et N-D, Transformer (single-head et multi-head 3D batched), GAN (1D, N-D, 3D volumétrique, RGB), LDM (diffusion conditionnelle image et audio), SLM (5 tâches NLP), Vision (classifieur d'images et de parole avec extraction de features), MLP (données tabulaires multi-classes).
- **API V3 unifiée** : Un point d'entrée `Trainer` pour entraîner tous les modèles avec callbacks Keras-like, sauvegarde automatique `.ia`, et chargement via `Trainer.load()`. Supporte le paramètre `dataset=` pour charger directement depuis un fichier ou un dossier.
- **Rétro-compatibilité V2** : Toutes les fonctions `train_xxx()` et `load_xxx()` / `predict_xxx()` restent disponibles comme raccourcis.
- **Format binaire natif `.ia`** : Format binaire simple (magic `IAV3`, header JSON, tenseurs bruts), sans dépendance, lisible depuis C via `fread`/`memcpy`.
- **Callbacks Keras-like** : `EarlyStopping`, `ModelCheckpoint`, `ProgressPrinter`, `CSVLogger`, et API pour créer des callbacks personnalisés.
- **Chargement de datasets multi-formats** : CSV, TSV, JSON, JSONL, NPY, NPZ, TXT, Parquet, HDF5, XLSX, Pickle, GZip, Images (JPG, PNG, BMP, WEBP, TIFF), Audio (WAV, FLAC, OGG, MP3) — détection automatique par extension, chargement par dossier avec labels automatiques.
- **Zero dépendance C++ externe** : Pas de CMake, pas de pybind11, pas de BLAS/LAPACK. Uniquement `g++` et `make`.
- **API zero-copy** (Phase 1) : Fonctions C++ opérant directement sur des pointeurs bruts (`relu_inplace`, `sigmoid_inplace`, `add_inplace`, `axpy_inplace`, `scale_inplace`, `matmul_2d_into`, `convolve2d_into`) pour les chemins critiques.
- **Configuration centralisée** : Toutes les constantes définies dans `config.py`, surchargeables via variables d'environnement `IA_*`.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    API Python (IA)                       │
├──────────┬──────────┬───────────┬───────────────────────┤
│  Trainer  │  Model   │ Callbacks  │  Dataset Loader      │
│  (V3)    │  (.ia)   │ (Keras)    │  (multi-formats)      │
├──────────┴──────────┴───────────┴───────────────────────┤
│              train/                infer/                 │
│  ┌─────┬─────┬──────┬─────┬─────┬─────┬────────┐       │
│  │ CNN │ RNN │Transf│ GAN │ LDM │ SLM │ Vision │       │
│  └──┬──┴──┬──┴──┬───┴──┬──┴──┬──┴──┬──┴───┬────┘       │
│     │     │     │      │     │     │      │             │
├─────┴─────┴─────┴──────┴─────┴─────┴──────┴─────────────┤
│              cpp/__init__.py (ctypes wrapper)            │
├─────────────────────────────────────────────────────────┤
│              _ia_core.so (C++ pur, C API)                │
│  ┌──────────────┬──────────────┬──────────────────┐     │
│  │  engine.h/.cpp│ autograd.h/.cpp│  c_api.cpp     │     │
│  │  Tensor, ops  │  Wengert tape  │  extern "C"    │     │
│  └──────────────┴──────────────┴──────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

**Principe** : Tous les calculs mathématiques sont délégués au moteur C++ via ctypes. Les modules Python (`train/`, `infer/`) ne font que de l'orchestration : ils construisent les graphes de calcul, gèrent les boucles d'entraînement et la rétropropagation, mais chaque opération élémentaire (matmul, convolution, activation, etc.) appelle une fonction C++.

## Structure du projet

```
ia_v3_source/
├── __init__.py          # Point d'entrée, exports publics, raccourcis V2
├── config.py            # Configuration centralisée (env vars IA_*)
├── trainer.py           # Classe Trainer V3 (API unifiée) + registre de modèles
├── model.py             # Classe Model (inférence + persistance .ia)
├── callbacks.py         # Callbacks Keras-like
├── dataset.py           # Chargement de datasets multi-formats
├── ia_format.py         # Format binaire natif .ia
├── setup.py             # Fallback setuptools (optionnel)
├── Makefile             # Compilation principale (_ia_core.so)
│
├── cpp/                 # Moteur de calcul C++
│   ├── __init__.py      # Wrapper ctypes (binder générique, fallback numpy)
│   ├── engine.h         # Déclarations : Tensor, ops, conv, norm, FFT...
│   ├── engine.cpp       # Implémentations C++ de toutes les opérations
│   ├── autograd.h       # Autograd minimaliste (Wengert tape)
│   ├── autograd.cpp     # Implémentation de l'autograd
│   └── c_api.cpp        # API C pure (extern "C") pour ctypes
│
├── train/               # Entraînement de tous les modèles
│   ├── __init__.py
│   ├── cnn.py           # CNN 2D et N-D
│   ├── rnn.py           # RNN vanilla (BPTT)
│   ├── transformer.py   # Transformer single-head + 3D multi-head
│   ├── gan.py           # GAN 1D, N-D, 3D, RGB
│   ├── ldm.py           # Diffusion conditionnelle (image, audio)
│   ├── slm.py           # SLM : next-word, emotion, mood, statement, sentiment
│   └── vision.py        # Classifieur d'images et de parole
│
├── infer/               # Inférence de tous les modèles
│   ├── __init__.py
│   ├── cnn.py           # load/predict CNN 2D et N-D
│   ├── rnn.py           # load/predict RNN
│   ├── transformer.py   # load/predict Transformer
│   ├── gan.py           # load/generate GAN (1D, N-D, 3D, RGB)
│   ├── ldm.py           # load/generate LDM (image, audio)
│   ├── slm.py           # load/predict SLM
│   └── vision.py        # load/predict image et speech classifier
│
└── models/              # Modèles pré-entraînés (.ia et .gy)
    ├── mini_rnn.ia / .gy
    ├── mini_cnn.ia / .gy
    ├── mini_transformer.ia / .gy
    ├── mini_gan.ia / .gy
    ├── mini_ldm.ia / .gy
    ├── mini_slm.ia / .gy
    └── mini_image_classifier.ia / .gy
```

## Prérequis et installation

### Dépendances système

| Outil | Version minimale | Usage |
|-------|-----------------|-------|
| **g++** | C++17 support | Compilation du moteur C++ |
| **make** | GNU Make | Build automatisé |
| **python** | >= 3.9 | Runtime Python |

### Dépendances Python

| Package | Requis ? | Usage |
|---------|----------|-------|
| **numpy** | Oui | Interface tableaux (seule vraie dépendance) |
| **pyarrow** | Optionnel | Lecture Parquet |
| **h5py** | Optionnel | Lecture HDF5 |
| **openpyxl** | Optionnel | Lecture Excel (.xlsx) |

### Compilation du moteur C++

```bash
cd ia_v3_source/
make
```

La compilation produit `cpp/_ia_core.so` (ou `_ia_core.cpython-3xx-linux-x86_64.so` selon la plateforme). Les flags de compilation sont :

- `-std=c++17` : Standard C++17
- `-O3 -march=native -ffast-math` : Optimisations agressives
- `-fPIC` : Position-independent code (shared library)
- `-fopenmp` : Support OpenMP pour le parallélisme

**Vérifier la compilation :**

```bash
make test
```

Sortie attendue :

```
matmul: [[19. 22.]
 [43. 50.]]
relu: [0.  0.5 2. ]
OK
```

**Nettoyer :**

```bash
make clean
```

### Installation alternative (pip)

Un `setup.py` est fourni comme fallback si vous préférez pip :

```bash
cd ia_v3_source/
pip install -e .
```

### Fallback numpy

Si `_ia_core.so` n'est pas trouvé (ex. pas encore compilé), le module bascule automatiquement en mode **fallback** où toutes les opérations C++ sont remplacées par des équivalents numpy. Un avertissement est affiché :

```
WARNING:IA.cpp:IA: _ia_core.so non trouve. Compilez avec: cd IA && make
```

Ce mode est fonctionnel mais moins performant. Il permet de développer et tester sans compiler le C++.

## Démarrage rapide

### API V3 unifiée (recommandée)

L'API V3 utilise la classe `Trainer` comme point d'entrée unique pour tous les modèles.

```python
from IA import Trainer

# Créer un trainer
trainer = Trainer(verbose=True)

# Entraîner un RNN
model = trainer.train(type='rnn', epochs=500, lr=0.01)

# Prédire
import numpy as np
X_new = np.array([[[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]])
y_pred = model.predict(X_new)
print(f"Prédiction : {y_pred}")

# Sauvegarder
model.save('mon_rnn.ia')

# Recharger plus tard
model2 = Trainer.load('mon_rnn.ia')
y_pred2 = model2.predict(X_new)
```

**Entraîner avec EarlyStopping et checkpoint :**

```python
from IA import Trainer
from IA.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

trainer = Trainer(verbose=True, callbacks=[
    ModelCheckpoint('best_cnn.ia', monitor='loss', save_best_only=True),
    CSVLogger('training_log.csv'),
])

model = trainer.train(
    type='cnn',
    epochs=2000,
    lr=0.01,
    early_stopping_patience=50,
)
```

**Types de modèles supportés par le Trainer :**

| `type` | Description | Données par défaut |
|--------|-------------|-------------------|
| `'rnn'` | RNN vanilla, classification binaire de séquences | 2 séquences, 5 pas de temps, 2 features |
| `'cnn'` | CNN 2D, classification de motifs (croix, carré, etc.) | 8 images 5x5 |
| `'transformer'` | Transformer single-head, classification de séquences | 8 séquences de 4 tokens, vocab=6 |
| `'gan'` | GAN N-D, génération de vecteurs 16D | Mélange de gaussiennes |
| `'ldm'` | Diffusion conditionnelle image 8x8 | 5 classes géométriques |
| `'slm'` | SLM prédiction du mot suivant | Vocabulaire français, 10 mots |
| `'image_classifier'` | Classifieur d'images (FC + softmax) | 3 classes synthétiques |
| `'mlp'` | MLP tabulaire multi-classes (ReLU + softmax/sigmoid) | 200 échantillons, 4 features |

### API V2 rétro-compatible

Toutes les fonctions d'entraînement et d'inférence V2 restent accessibles directement :

```python
# Entraînement
from IA import train_rnn, train_cnn2d, train_gan_nd, train_slm_emotion

result_rnn = train_rnn(epochs=500, lr=0.01)
result_cnn = train_cnn2d(epochs=1000, lr=0.01)
result_gan = train_gan_nd(data_dim=16, latent_dim=16, epochs=5000)
result_slm = train_slm_emotion(epochs=500, lr=0.01)

# Inférence
from IA import load_rnn, predict_rnn, load_gan_nd, generate_gan_nd

model_rnn = load_rnn('models/mini_rnn.gy')
pred = predict_rnn(model_rnn, sequence)
# -> {'probability': 0.92, 'class': 'Classe 1', 'confidence': 0.84, ...}

model_gan = load_gan_nd('models/mini_gan.gy')
result = generate_gan_nd(model_gan, num_samples=10)
# -> {'generated': ndarray(10, 16), 'latent_vectors': ndarray(10, 16)}
```

## Modèles supportés

### RNN (Réseau de Neurones Récurrent)

**Architecture** : RNN vanilla avec mécanisme de rétropropagation dans le temps (BPTT). Activation tanh pour l'état caché, sigmoïde pour la sortie.

**Poids** : `W_xh` (entrée→caché), `W_hh` (caché→caché), `b_h` (biais caché), `W_hy` (caché→sortie), `b_y` (biais sortie).

```python
# V3
model = Trainer().train(type='rnn', epochs=500, lr=0.01)

# V2
from IA import train_rnn, load_rnn, predict_rnn
result = train_rnn(X, y, hidden_size=8, lr=0.01, epochs=1000, grad_clip=1.0)
model = load_rnn('models/mini_rnn.gy')
pred = predict_rnn(model, sequence)  # sequence: (seq_len, input_size)
```

**Retour `predict_rnn`** :

```python
{
    'probability': 0.92,     # Probabilité de la classe 1
    'class': 'Classe 1',     # Classe prédite
    'confidence': 0.84,      # |probability - 0.5| * 2
    'hidden_states': [...]   # Liste des états cachés à chaque pas
}
```

### CNN (Réseau de Neurones Convolutif)

Deux variantes sont disponibles :

#### CNN 2D

Classification d'images 2D avec motifs synthétiques (croix, carré, diagonale, bordure, point, aléatoire). Architecture : Conv2D → ReLU → Flatten → FC → ReLU.

```python
# V3
model = Trainer().train(type='cnn', epochs=1000, lr=0.01)

# V2
from IA import train_cnn2d, load_cnn2d, predict_cnn2d
result = train_cnn2d(input_shape=(5,5), kernel_shape=(3,3), lr=0.01, epochs=1000)
model = load_cnn2d('models/mini_cnn.gy')
pred = predict_cnn2d(model, image)  # image: (H, W)
```

#### CNN N-D

Classification de volumes N-dimensionnels (configurable de 2D à 5D+). Même architecture que le CNN 2D mais avec convolution N-D générique et rétropropagation correspondante.

```python
from IA import train_cnn_nd, load_cnn_nd, predict_cnn_nd
result = train_cnn_nd(dimensions=4, volume_shape=(3,3,3,3), kernel_shape=(2,2,2,2))
```

### Transformer

Deux implémentations sont fournies :

#### MiniTransformer (single-head)

Classification de séquences de tokens avec self-attention, résiduels, layer normalization et feed-forward. Embedding appris conjointement.

```python
# V3
model = Trainer().train(type='transformer', epochs=2000, lr=0.01)

# V2
from IA import train_transformer, load_transformer, predict_transformer
result = train_transformer(
    seq_len=4, vocab_size=6, embed_dim=8, ff_dim=16,
    lr=0.01, epochs=2000
)
```

**Architecture** : Embedding → Q/K/V → Attention → Résiduel + LayerNorm → FFN (2 couches) → Résiduel + LayerNorm → Mean Pooling → Classification.

#### MiniTransformer3D (multi-head batched)

Version avancée avec attention multi-têtes et support de batch. Implémentée comme une classe Python (`MiniTransformer3D`) avec méthodes `forward()`, `backward()`, `save()`, `load()`.

```python
from IA import train_transformer3d
result = train_transformer3d(
    vocab_size=6, seq_len=4, embed_dim=8, ff_dim=16,
    num_heads=2, lr=0.01, epochs=5000
)
```

### GAN (Réseau Génératif Adversarial)

Quatre variantes de GAN sont implémentées, toutes partageant une boucle d'entraînement commune (`_train_gan_core`) avec un discriminateur entraîné 2 fois par étape de générateur et loss non-saturating.

#### GAN 1D

Génère des valeurs scalaires à partir d'un mélange de 2 gaussiennes N(-2, 0.5) et N(+2, 0.5). Pas de tanh sur la sortie.

```python
# V3
model = Trainer().train(type='gan', epochs=5000, lr=0.001)

# V2
from IA import train_gan_1d, load_gan_1d, generate_gan_1d
result = train_gan_1d(latent_dim=2, hidden_dim=16, lr=0.001, epochs=5000, batch_size=32)
model = load_gan_1d('models/gan_1d.gy')
gen = generate_gan_1d(model, num_samples=100)
# gen['generated']: (100, 1)
```

#### GAN N-D

Génère des vecteurs de dimension `data_dim` (défaut 16). Tanh sur la sortie.

```python
from IA import train_gan_nd, load_gan_nd, generate_gan_nd
result = train_gan_nd(data_dim=16, latent_dim=16, hidden_dim=64, epochs=5000)
gen = generate_gan_nd(model, num_samples=50)
# gen['generated']: (50, 16)
```

#### GAN 3D

Génère des volumes 4x4x4 (64 voxels) représentant des sphères ou des cubes avec bruit. Tanh sur la sortie.

```python
from IA import train_gan_3d, load_gan_3d, generate_gan_3d
result = train_gan_3d(volume_size=4, latent_dim=16, hidden_dim=128, lr=0.0005, epochs=5000)
gen = generate_gan_3d(model)
# gen['generated_volume']: (4, 4, 4)
# gen['volume_shape']: (4, 4, 4)
```

#### GAN RGB

Génère des images 32x32x3 (3072 pixels) avec des motifs de couleur synthétiques (dégradés, formes géométriques, rayures, bruit, mixte).

```python
from IA import train_gan_rgb, load_gan_rgb, generate_gan_rgb
result = train_gan_rgb(
    image_size=32, channels=3, latent_dim=200, hidden_dim=512,
    lr=0.00002, epochs=10, batch_size=16
)
gen = generate_gan_rgb(model, num_samples=5)
# gen['generated_images']: (5, 32, 32, 3)
# gen['image_shape']: (32, 32, 3)
```

**Architecture commune GAN** :

- **Générateur** (3 couches) : `latent → LeakyReLU(hidden) → LeakyReLU(hidden) → tanh/linear(output)`
- **Discriminateur** (3 couches) : `input → LeakyReLU(hidden) → LeakyReLU(hidden) → sigmoid(1)`

### LDM (Modèle de Diffusion Latente)

Deux variantes de modèles de diffusion conditionnelle (DDPM) sont implémentées, partageant la même boucle d'entraînement (`_train_ldm_core`) avec un schedule beta linéaire de 1e-4 à 0.02.

#### LDM Image

Génération conditionnelle d'images 2D 8x8 représentant des formes géométriques : rectangle (0), cercle (1), triangle (2), croix (3), ellipse (4).

```python
# V3
model = Trainer().train(type='ldm', epochs=1000, lr=0.001)

# V2
from IA import train_ldm_image, load_ldm_image, generate_ldm_image
result = train_ldm_image(
    image_size=8, num_classes=5, timesteps=200, lr=0.001, epochs=1000
)
model = load_ldm_image('models/ldm_image.gy')
gen = generate_ldm_image(model, class_id=1, shape=(8, 8), num_steps=50)
# gen['generated']: (8, 8) ndarray
# gen['class_id']: 1
# gen['num_steps']: 50
```

#### LDM Audio

Génération conditionnelle de signaux audio 1D (64 samples) : sinusoïde (0), onde carrée (1), dent de scie (2), bruit (3), chirp (4).

```python
from IA import train_ldm_audio, load_ldm_audio, generate_ldm_audio
result = train_ldm_audio(
    signal_length=64, num_classes=5, timesteps=200, lr=0.001, epochs=1000
)
gen = generate_ldm_audio(model, class_id=0, signal_length=64, num_steps=50)
# gen['generated']: (64,) ndarray
```

**Architecture LDM** : Le réseau de prédiction de bruit est un `SimpleDiffusionNet` avec :
- Embedding de classe appris `(num_classes → hidden_dim)`
- FC1 : `(input_dim + hidden_dim) → hidden_dim` + ReLU
- FC2 : `hidden_dim → input_dim`

La génération utilise le processus de débruitage DDPM itératif (reverse diffusion).

### SLM (Small Language Model)

Cinq tâches NLP sont implémentées, toutes basées sur la classe `SLMClassifier` avec une architecture transformer (embedding → self-attention → LayerNorm → FFN avec GELU → LayerNorm → pooling → classification) et un préprocesseur texte `TextPreprocessor`.

| Fonction | Tâche | Classes |
|----------|-------|---------|
| `train_slm_next_word` | Prédiction du mot suivant | Vocabulaire français (10 mots) |
| `train_slm_emotion` | Détection d'émotion | joie, tristesse, colère, peur, surprise, neutre (6) |
| `train_slm_mood` | Détection d'humeur | joyeux, triste, énergique, calmé, stressé, fatigué, motivé, anxieux (8) |
| `train_slm_statement` | Classification de type de phrase | question, affirmation, ordre, conseil, exclamation (5) |
| `train_slm_sentiment` | Analyse de sentiment | positif, négatif, neutre (3) |

```python
# V3
model = Trainer().train(type='slm', epochs=500, lr=0.01)

# V2 — Entraînement
from IA import train_slm_emotion, train_slm_sentiment
result = train_slm_emotion(epochs=500, lr=0.01)
result = train_slm_sentiment(epochs=500, lr=0.01)

# V2 — Inférence
from IA import load_slm, predict_slm, predict_slm_next_word
model = load_slm('models/mini_slm.gy')
pred = predict_slm(model, "Je suis très heureux aujourd'hui")
# -> {
#      'predictions': [
#          {'class': 'joie', 'probability': 0.85},
#          {'class': 'neutre', 'probability': 0.10},
#          ...
#      ],
#      'attention_weights': ndarray(seq_len, seq_len)
#  }

next_words = predict_slm_next_word(model, [3, 7, 1])
# -> {'top_words': [{'word': 'bonjour', 'probability': 0.25}, ...]}
```

### MLP (Perceptron Multi-Couches)

Architecture dense entièrement connectée pour données tabulaires. Couches cachées avec activation ReLU, sortie sigmoid (binaire) ou softmax (multi-classes). Initialisation He, mini-batch SGD, détection automatique du nombre de classes.

```python
# V3 — données démo
model = Trainer().train(type='mlp', epochs=500, lr=0.01)

# V3 — données réelles (CSV)
from IA.dataset import load_dataset
X, y = load_dataset("diabetes.csv", target="Diabetes_012")
model = Trainer().train(
    type='mlp', X=X, y=y,
    hidden_sizes=[64, 32], epochs=500, lr=0.01,
    save_path="/chemin/vers/diabetes_mlp"
)

# V3 — chargement direct depuis fichier
model = Trainer().train(
    type='mlp',
    dataset="diabetes.csv", dataset_target="Diabetes_012",
    hidden_sizes=[64, 32], epochs=500
)

# V2
from IA.train.mlp import train_mlp
result = train_mlp(X=X, y=y, hidden_sizes=[64, 32], epochs=500, lr=0.01)
```

**Paramètres spécifiques :**

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `hidden_sizes` | `[64, 32]` | Taille de chaque couche cachée |
| `batch_size` | `256` | Taille de mini-batch |
| `dropout` | `0.0` | Taux de dropout entre couches cachées |

**Architecture** : `input → [Dense(n) → ReLU] × L → Dense(n_out) → sigmoid/softmax`

**Quand utiliser** : données tabulaires (CSV, Excel, JSON), features numériques sans structure spatiale ou temporelle. Préférer `rnn` pour les séquences, `cnn` ou `image_classifier` pour les images.

### Vision (Classifieur d'images et de parole)

#### Classifieur d'images

Classification d'images via extraction de features (histogramme couleur RGB, gradient Sobel, LBP, détection de bords) suivie d'un réseau FC à 2 couches avec softmax.

```python
# V3
model = Trainer().train(type='image_classifier', epochs=500, lr=0.01)

# V2
from IA import train_image_classifier, load_image_classifier, predict_image
result = train_image_classifier(
    num_classes=3, feature_dim=128, hidden_dim=64, lr=0.01, epochs=500
)
model = load_image_classifier('models/mini_image_classifier.gy')
pred = predict_image(model, image)  # image: (H, W) ou (H, W, 3)
# -> {'class': 'classe_A', 'confidence': 0.92, 'probabilities': {...}, 'feature_vector': ...}
```

**Features extraites (128 dim)** :
- Histogramme couleur RGB normalisé (16 bins × 3 canaux = 48 features)
- Magnitude de gradient moyenne (1 feature)
- Texture LBP simplifiée (16 bins = 16 features)
- Détection de bords sur grille 4×4 (16 features)
- Padding à 128 dim si nécessaire

#### Classifieur de parole

Classification de phonèmes (voyelle, consonne, silence) via extraction de features audio (MFCC simplifié avec filtres mel, delta, DCT-II) suivie d'une convolution 1D temporelle avec dilation et d'un réseau FC.

```python
from IA import train_speech_classifier, load_speech_classifier, predict_speech
result = train_speech_classifier(
    num_classes=3, feature_dim=26, hidden_dim=64, lr=0.01, epochs=500
)
model = load_speech_classifier('models/speech_classifier.gy')
pred = predict_speech(model, audio_signal)  # audio_signal: (n_samples,)
# -> {'class': 'voyelle', 'confidence': 0.88, 'probabilities': {...}, 'feature_vector': ...}
```

**Architecture** : `signal → MFCC (13 coeffs × n_frames) + delta (13) → Conv1D(in=1, out=64, k=3) → ReLU → Global Avg Pool → FC(64→3) → Softmax`.

## Moteur de calcul C++

### Opérations disponibles

Le fichier `engine.h` définit l'ensemble des opérations dans le namespace `ia_core` :

#### 1. Types et utilitaires bas niveau

| Fonction | Description |
|----------|-------------|
| `Tensor` | Classe légère : `vector<double>` + `Shape` (forme ND) |
| `Shape` | Structure de forme avec `dims`, `size()`, `ndim()` |
| `tensor.reshape()` | Reshape sans copie (vérifie la taille) |
| `tensor.flatten()` | Aplatissement en 1D |
| `tensor.transpose2d()` | Transposition 2D |
| `tensor.rows(start, end)` | Slice de lignes |
| `tensor.copy()` | Copie profonde |

#### 2. Fonctions d'activation

| Fonction | Description |
|----------|-------------|
| `relu(x)` / `relu_deriv(x)` | ReLU et sa dérivée |
| `sigmoid(x)` / `sigmoid_deriv(x)` | Sigmoïde et sa dérivée |
| `tanh_act(x)` / `tanh_deriv(x)` | Tanh et sa dérivée |
| `leaky_relu(x, alpha)` / `leaky_relu_deriv(x, alpha)` | Leaky ReLU (α=0.01) |
| `softmax(x)` | Softmax sur le dernier axe |
| `gelu(x)` / `gelu_deriv(x)` | GELU et sa dérivée |

#### 3. Algèbre linéaire

| Fonction | Description |
|----------|-------------|
| `matmul(A, B)` | Multiplication matricielle (2D et 3D batched) |
| `mul(A, B)` | Produit élément-wise avec broadcasting basique |
| `add(A, B)` / `sub(A, B)` / `div(A, B)` | Opérations élément-wise |
| `outer(a, B)` | Produit extérieur |
| `scale(x, s)` | Scalaire × Tensor |
| `add_scalar(x, s)` | Tensor + scalaire |
| `clip(x, lo, hi)` | Clip entre lo et hi |
| `exp(x)` / `log(x)` / `sqrt(x)` / `pow(x, n)` | Opérations élément-wise |
| `abs(x)` / `sign(x)` / `neg(x)` | Valeur absolue, signe, négation |
| `maximum(A, B)` | Maximum élément-wise |
| `concatenate(A, B)` | Concaténation sur l'axe 0 |
| `tile(x, n)` / `repeat(x, n)` | Répétition de tenseurs |

#### 4. Convolutions

| Fonction | Description |
|----------|-------------|
| `convolve2d(img, kernel)` | Convolution 2D : `(ih,iw) × (kh,kw) → (oh,ow)` |
| `convolve2d_backward(img, kernel, d_conv)` | Rétropropagation convolution 2D |
| `conv1d_forward(x, W, b, k, d)` | Convolution 1D avec dilation |
| `conv1d_backward(x, W, d_out, k, d)` | Rétropropagation convolution 1D (retourne d_x, d_W, d_b) |
| `convolve_nd(volume, kernel)` | Convolution N-D générique (récursive) |
| `convolve_nd_backward(volume, kernel, d_conv)` | Rétropropagation convolution N-D |

#### 5. Normalisation

| Fonction | Description |
|----------|-------------|
| `layer_norm(x, eps)` | Layer normalization sur le dernier axe |

#### 6. Fonctions de perte

| Fonction | Description |
|----------|-------------|
| `mse_loss(pred, target)` | MSE : `mean((pred - target)²)` |
| `mse_loss_grad(pred, target)` | Gradient MSE : `2*(pred - target) / n` |
| `cross_entropy_loss(logits, target_idx)` | Cross-entropy : `-log(soft[logits][target_idx])` |

#### 7. Initialisation

| Fonction | Description |
|----------|-------------|
| `zeros(shape)` / `ones(shape)` | Tenseurs remplis de 0 ou 1 |
| `xavier_init(shape, seed)` | Xavier/Glorot : `randn * sqrt(2 / sum(shape))` |
| `randn(shape, seed)` | Distribution normale N(0,1) |
| `uniform(shape, lo, hi, seed)` | Distribution uniforme |
| `permutation(n, seed)` | Permutation aléatoire de [0..n-1] |
| `randint(lo, hi, seed)` | Entier aléatoire dans [lo, hi) |

#### 8. Réductions

| Fonction | Description |
|----------|-------------|
| `sum(x)` / `sum_axis(x, axis)` | Somme totale ou par axe |
| `mean(x)` / `mean_axis(x, axis)` | Moyenne totale ou par axe |
| `var_axis(x, axis)` | Variance par axe |
| `max_val(x)` / `max_axis(x)` | Maximum total ou par axe |
| `argmax(x)` | Indice du maximum sur le dernier axe |
| `histogram(x, n_bins, lo, hi)` | Histogramme |

#### 9. FFT et opérations spécifiques

| Fonction | Description |
|----------|-------------|
| `fft_rfft(x, n_fft)` | FFT 1D (Cooley-Tukey radix-2) |
| `add_noise(x0, sqrt_alpha, sqrt_one_minus)` | Ajout de bruit de diffusion |
| `linear(x, W, b)` | Couche linéaire : `x @ W + b` |
| `dot1d(a, b)` | Produit scalaire 1D |
| `ldm_predict_noise(...)` | Forward LDM complet |
| `pad1d(x, target_len)` | Padding 1D |
| `diff_axis(x, axis)` | Différence absolue par axe |
| `linspace(start, end, num)` | Linéar spacing |
| `arange(start, end, step)` | Arange |
| `mgrid3d(size)` | Grille 3D |
| `gather(x, indices)` / `scatter(x, idx, val)` | Indexation |
| `dct2(x, num_coeffs)` | DCT-II simplifiée |

### API zero-copy

Pour les chemins critiques, des fonctions opèrent directement sur des pointeurs bruts sans allocation intermédiaire :

```c
void relu_inplace(double* x, int64_t n);
void sigmoid_inplace(double* x, int64_t n);
void tanh_inplace(double* x, int64_t n);
void add_inplace(double* a, const double* b, int64_t n);
void axpy_inplace(double* a, const double* b, int64_t n, double s);
void scale_inplace(double* a, int64_t n, double s);
void matmul_2d_into(const double* A, int64_t M, int64_t K,
                    const double* B, int64_t N, double* C);
void convolve2d_into(const double* img, int64_t ih, int64_t iw,
                     const double* kernel, int64_t kh, int64_t kw,
                     double* out);
```

### Autograd (Wengert Tape)

Un système de différenciation automatique minimaliste est implémenté dans `autograd.h/.cpp`. Il utilise un **Wengert tape** (style micrograd/tinygrad) global et thread-local.

**Portée actuelle** : `relu`, `sigmoid`, `tanh`, `add`, `sub`, `mul`, `scale`, `matmul`, `mse`, `dot`.

```python
# Utilisation depuis Python (via l'API C)
from IA.cpp import get_core
C = get_core()

# L'autograd est accessible via les fonctions ag_*
# (exposées dans c_api.cpp)
```

> **Note** : L'autograd est un prototype (Phase 4). Les opérations plus complexes (conv, softmax, layer_norm) restent à ajouter. L'API est extensible sans casser le code existant.

## Format binaire `.ia`

Le format `.ia` est un format binaire natif pour la sérialisation des modèles, conçu pour être simple, portable et lisible depuis n'importe quel langage (C, C++, Python, Rust, etc.) sans bibliothèque externe.

### Spécification binaire

```
Offset  Taille  Champ
0       4       Magic : "IAV3" (4 bytes)
4       2       Version majeure (uint16 LE)
6       2       Version mineure (uint16 LE)
8       4       Taille du header JSON (uint32 LE)
12      N       Header JSON (UTF-8) : type, config, weights_meta, ...
12+N    4       Nombre de tenseurs (uint32 LE)
16+N    ...     Pour chaque tenseur :
                  - 1 byte  : dtype (0=f32, 1=f64, 2=i64)
                  - 4 bytes : ndim (uint32 LE)
                  - 8*ndim bytes : shape (int64 LE)
                  - 4 bytes : taille du nom (uint32 LE)
                  - M bytes : nom (UTF-8)
                  - prod(shape)*sizeof(dtype) bytes : données brutes
```

### API Python

```python
from IA import ia_format

# Sauvegarder
ia_format.save_model(
    path='modele.ia',
    header={'type': 'rnn', 'config': {'hidden_size': 4}},
    tensors={'W_xh': np.array([[1,2],[3,4]]), 'b_h': np.zeros(4)}
)

# Charger
header, tensors = ia_format.load_model('modele.ia')
# header: {'type': 'rnn', 'config': {'hidden_size': 4}, ...}
# tensors: {'W_xh': ndarray, 'b_h': ndarray}

# Infos sans charger les tenseurs
info = ia_format.model_info('modele.ia')
```

### Avantages

- **Zéro dépendance** : Lecture/écriture avec seulement `struct` et `json` (standard Python) ou `fread`/`fwrite` (C)
- **Portable** : Format little-endian, types de base (f32, f64, i64)
- **Léger** : Pas de compression, pas de métadonnées superflues
- **Compatible C** : Peut être lu directement en C via `memcpy`/`fread`

## Chargement de datasets

### Formats supportés

| Extension | Format | Dépendance |
|-----------|--------|------------|
| `.csv` | CSV (virgule) | Aucune |
| `.tsv` | TSV (tabulation) | Aucune |
| `.json` | JSON (liste d'objets ou dict) | Aucune |
| `.jsonl` / `.ndjson` | JSON Lines (un objet par ligne) | Aucune |
| `.npy` | NumPy array | numpy |
| `.npz` | NumPy zipped archive | numpy |
| `.txt` | Texte brut (numérique ou NLP) | Aucune |
| `.h5` / `.hdf5` | HDF5 | h5py |
| `.parquet` / `.pq` | Parquet | pyarrow |
| `.xlsx` / `.xls` | Excel | openpyxl / xlrd |
| `.pkl` / `.pickle` | Pickle | Aucune |
| `.gz` | GZip (délègue au bon loader) | Aucune |
| `.jpg` `.jpeg` `.png` `.bmp` `.gif` `.webp` `.tiff` | Image → `ndarray (H, W, C)` float64 [0-1] | Pillow |
| `.wav` `.flac` `.ogg` `.mp3` | Audio → `(signal_1D, sample_rate)` | soundfile / pydub / scipy |

### API

```python
from IA.dataset import load_dataset, supported_formats, dataset_info

# Charger un CSV avec colonne cible
X, y = load_dataset("donnees.csv", target="label")
# X: ndarray(N, n_features), y: ndarray(N, 1)

# Charger un JSON
X, y = load_dataset("donnees.json", target="classe")

# Charger un fichier numpy (pas de target)
X = load_dataset("features.npy")

# Charger du texte brut (pour SLM/NLP)
lines, _ = load_dataset("texte.txt")  # retourne ndarray d'objets

# Charger du Parquet
X, y = load_dataset("data.parquet", target="target")

# Charger une image → ndarray (H, W, C) float64 normalisé [0-1]
arr, _ = load_dataset("photo.jpg")
arr, _ = load_dataset("photo.jpg", image_size=(64, 64), mode="grayscale")

# Charger un fichier audio → (signal_1D float64, sample_rate)
sig, sr = load_dataset("son.wav")
sig, sr = load_dataset("son.mp3", sr=16000, max_len=16000)

# Lister les formats supportés
print(supported_formats())
# ['.csv', '.h5', '.hdf5', '.jpg', '.jpeg', '.json', '.jsonl', '.mp3',
#  '.ndjson', '.npy', '.npz', '.ogg', '.parquet', '.pkl', '.pickle',
#  '.pkl.gz', '.png', '.pq', '.tsv', '.txt', '.wav', '.xlsx', '.xls', '.gz']
```

#### Chargement par dossier (images ou audio)

```python
from IA.dataset import load_image_dataset, load_audio_dataset, load_folder

# Dossier d'images organisé par classe :
# images/chat/img1.jpg, images/chien/img2.jpg ...
X, y, class_names = load_image_dataset("images/", image_size=(64, 64))
# X: (N, 64*64*3), y: (N, 1) int64, class_names: ['chat', 'chien']

# Options image
X, y, classes = load_image_dataset(
    "images/", image_size=(128, 128), mode="grayscale", normalize=True
)

# Dossier audio organisé par classe :
# audio/bonjour/clip1.wav, audio/merci/clip2.wav ...
X, y, class_names = load_audio_dataset("audio/", sr=16000, max_len=16000)
# X: (N, 16000), y: (N, 1) int64

# Via Trainer directement (paramètre dataset=)
from IA import Trainer
trainer = Trainer(verbose=True)
trainer.train(type="mlp", dataset="donnees.csv", dataset_target="label", epochs=500)
trainer.train(type="mlp", dataset="images/", epochs=100)   # dossier images
trainer.train(type="mlp", dataset="audio/",  epochs=100)   # dossier audio
```

### Informations sur un dataset

```python
from IA.dataset import dataset_info

info = dataset_info("donnees.csv")
# {
#     'format': '.csv',
#     'size_bytes': 12345,
#     'n_rows': 1000,
#     'n_cols': 10,
#     'has_header': True,
#     'columns': ['feature1', 'feature2', ..., 'label']
# }

info = dataset_info("photo.jpg")
# {
#     'format': '.jpg',
#     'size_bytes': 54321,
#     'width': 640, 'height': 480,
#     'n_channels': 3, 'mode': 'RGB',
#     'shape': (480, 640, 3)
# }

info = dataset_info("son.wav")
# {
#     'format': '.wav',
#     'size_bytes': 88200,
#     'sample_rate': 44100,
#     'n_channels': 1,
#     'n_frames': 44100,
#     'duration_sec': 1.0
# }
```

## Système de callbacks

Le système de callbacks suit l'API de Keras/TensorFlow. Le `Trainer` appelle les callbacks dans cet ordre à chaque étape de l'entraînement :

1. `on_train_begin(trainer)`
2. `on_epoch_begin(epoch, trainer)` (pour chaque époque)
3. `on_epoch_end(epoch, metrics, trainer)` — `metrics` = `{'loss': ..., 'accuracy': ...}`
4. `on_train_end(trainer)`

Un callback peut interrompre l'entraînement prématurément en positionnant `trainer.stop_training = True`.

### Callback (base)

```python
from IA.callbacks import Callback

class MyCallback(Callback):
    def on_train_begin(self, trainer):
        print(f"Début entraînement {trainer.model_type}")

    def on_epoch_end(self, epoch, metrics, trainer):
        if metrics.get('loss', 0) < 0.01:
            trainer.stop_training = True

    def on_train_end(self, trainer):
        print(f"Fin — modèle sauvegardé : {trainer.model_path}")
```

### EarlyStopping

Arrête l'entraînement quand une métrique cesse de s'améliorer pendant un nombre donné d'époques (patience).

```python
from IA.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='loss',       # Métrique à surveiller
    patience=10,          # Époques sans amélioration avant arrêt
    min_delta=0.0,        # Seuil minimum de variation
    mode='min'            # 'min' (baisse = amélioration) ou 'max'
)
```

### ModelCheckpoint

Sauvegarde le modèle au format `.ia` à chaque amélioration de la métrique surveillée.

```python
from IA.callbacks import ModelCheckpoint

mc = ModelCheckpoint(
    filepath='best_model.ia',
    monitor='loss',
    save_best_only=True,  # Ne sauvegarde que si amélioration
    mode='min'
)
```

### ProgressPrinter

Affiche la progression à intervalle régulier dans la console.

```python
from IA.callbacks import ProgressPrinter

pp = ProgressPrinter(
    interval=10,                      # Affiche toutes les 10 époques
    show_metrics=['loss', 'accuracy']  # Métriques à afficher
)
```

Sortie :

```
[Trainer] Démarrage entraînement : rnn, epochs=1000, lr=0.01
[Trainer] epoch    0/1000  loss=0.5234  accuracy=0.5000  (0.3s)
[Trainer] epoch   10/1000  loss=0.3121  accuracy=0.6000  (1.2s)
...
[Trainer] Terminé en 15.3s   →  models/mini_rnn.ia
```

### CSVLogger

Journalise les métriques dans un fichier CSV.

```python
from IA.callbacks import CSVLogger

cl = CSVLogger(
    filename='training_log.csv',
    append=False  # False = écraser, True = ajouter
)
```

Fichier produit :

```csv
epoch,loss,accuracy
0,0.5234,0.5
1,0.4891,0.55
2,0.4512,0.6
...
```

### Callbacks personnalisés

Créez un callback en héritant de `Callback` et en surchargeant les méthodes souhaitées :

```python
from IA.callbacks import Callback

class LossThreshold(Callback):
    """Arrête l'entraînement si la loss descend sous un seuil."""

    def __init__(self, threshold=0.001):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, metrics, trainer):
        if metrics.get('loss', float('inf')) < self.threshold:
            print(f"Seuil atteint à l'epoch {epoch}!")
            trainer.stop_training = True

# Utilisation
trainer = Trainer(callbacks=[LossThreshold(threshold=0.005)])
model = trainer.train(type='cnn', epochs=5000)
```

## Configuration

Toutes les constantes sont centralisées dans `config.py` et peuvent être surchargées via des **variables d'environnement** préfixées par `IA_` :

| Variable d'environnement | Constante | Défaut | Description |
|--------------------------|-----------|--------|-------------|
| `IA_LR` | `TRAIN_LR` | `0.01` | Taux d'apprentissage global |
| `IA_EPOCHS` | `TRAIN_EPOCHS` | `1000` | Nombre d'époques global |
| `IA_BATCH_SIZE` | `TRAIN_BATCH_SIZE` | `32` | Taille de batch |
| `IA_SEED` | `TRAIN_SEED` | `42` | Graine aléatoire |
| `IA_EARLY_STOP_LOSS` | `TRAIN_EARLY_STOP_LOSS` | `0.001` | Seuil de loss pour arrêt précoce |
| `IA_GRADIENT_CLIP` | `TRAIN_GRADIENT_CLIP` | `1.0` | Valeur maximale des gradients |
| `IA_D_STEPS_PER_G` | `TRAIN_D_STEPS_PER_G` | `2` | Pas de discriminateur par pas de générateur (GAN) |
| `IA_SAVE_BEST_ONLY` | `MODEL_SAVE_BEST_ONLY` | `true` | Ne sauvegarder que le meilleur modèle |

**Exemple :**

```bash
export IA_LR=0.001
export IA_EPOCHS=5000
export IA_SEED=123
python mon_entrainement.py
```

**Accès programmatique :**

```python
from IA.config import get_config, ensure_directories, MODELS_DIR, MODEL_EXTENSION

config = get_config(overrides={'TRAIN_LR': 0.005})
print(config)
# {'TRAIN_LR': 0.005, 'TRAIN_EPOCHS': 1000, 'MODELS_DIR': '/path/to/IA/models', ...}

ensure_directories()  # Crée le répertoire models/ si nécessaire
```

## Modèles pré-entraînés inclus

Le répertoire `models/` contient des modèles pré-entraînés dans deux formats :

| Fichier | Format | Description |
|---------|--------|-------------|
| `mini_rnn.ia` / `.gy` | `.ia` (binaire) / `.gy` (pickle) | RNN vanilla |
| `mini_cnn.ia` / `.gy` | `.ia` / `.gy` | CNN 2D 5×5 |
| `mini_transformer.ia` / `.gy` | `.ia` / `.gy` | Transformer single-head |
| `mini_gan.ia` / `.gy` | `.ia` / `.gy` | GAN N-D |
| `mini_ldm.ia` / `.gy` | `.ia` / `.gy` | LDM image 8×8 |
| `mini_slm.ia` / `.gy` | `.ia` / `.gy` | SLM next-word |
| `mini_image_classifier.ia` / `.gy` | `.ia` / `.gy` | Classifieur d'images |

- **`.ia`** : Format binaire natif (portable, rapide, lisible en C)
- **`.gy`** : Format pickle Python (rétro-compatibilité V2)

## Extension du framework

### Ajouter un nouveau type de modèle

1. **Créer le module d'entraînement** dans `train/mon_modele.py` :

```python
def train_mon_modele(X=None, y=None, lr=0.01, epochs=1000, save_path=None, seed=42):
    """Entraîne un nouveau modèle."""
    # ... boucle d'entraînement ...
    return {'model': params, 'save_path': save_path, 'accuracy': accuracy, 'history': history}
```

2. **Enregistrer dans `trainer.py`** en ajoutant une fonction de prédiction et d'extraction :

```python
@register_predict_fn('mon_modele')
def _predict_mon_modele(weights, config, X):
    C = _cpp()
    # ... forward pass ...
    return y_pred

def _extract_mon_modele(result):
    m = result['model']
    return ({k: m[k] for k in ['W1', 'b1', 'W2', 'b2']},
            {'hidden_dim': m['hidden_dim'], 'input_dim': m['input_dim']})

# Dans _register_all() :
register_model_type('mon_modele', train_mon_modele, _predict_mon_modele,
                    _extract_mon_modele, _extract_mon_modele, 'mini_mon_modele')
```

3. **Exporter** dans `__init__.py` et `train/__init__.py`.

### Ajouter une opération C++

1. Déclarer la fonction dans `cpp/engine.h` (namespace `ia_core`)
2. Implémenter dans `cpp/engine.cpp`
3. Exposer dans `cpp/c_api.cpp` (fonction `extern "C"`)
4. Binder dans `cpp/__init__.py` via `_bind()`

## Limitations connues

- **Pas de GPU** : Le moteur C++ est purement CPU. Pas de support CUDA/OpenCL.
- **Pas de batched training** (sauf Transformer3D) : La plupart des modèles itèrent échantillon par échantillon.
- **Pas de DataLoader** : Pas de chargement par batch avec shuffling automatique.
- **Autograd limité** : Seules les opérations de base ont une différenciation automatique. La rétropropagation est principalement écrite à la main.
- **Taille des modèles** : Les modèles sont de démonstration (mini), pas optimisés pour des tâches réelles à grande échelle.
- **Pas de pooling** : Pas d'implémentation de max/avg pooling dans le moteur C++ (le CNN flatten directement après la convolution).
- **Format .gy** : Le format pickle `.gy` est une dépendance Python qui n'est pas portable vers d'autres langages. Préférez le format `.ia`.

## Changelog V2 → V3

| Fonctionnalité | V2 | V3 |
|---------------|-----|-----|
| API d'entraînement | Fonctions séparées `train_xxx()` | `Trainer.train(type=...)` unifié |
| Persistance | Pickle `.gy` uniquement | Binaire natif `.ia` + pickle `.gy` (rétro-compat) |
| Inférence | Fonctions séparées `load_xxx()` / `predict_xxx()` | `Model.predict()` + `Trainer.load()` |
| Callbacks | Aucun | EarlyStopping, ModelCheckpoint, ProgressPrinter, CSVLogger |
| Format binaire | Aucun | `.ia` (IAV3, header JSON + tenseurs bruts) |
| Chargement de données | Aucun | Multi-formats (CSV, JSON, Parquet, HDF5, etc.) |
| Autograd | Aucun | Wengert tape (Phase 4, prototype) |
| API zero-copy | Aucune | Fonctions inplace/into pour les hot paths |
| Configuration | Hardcodée | Centralisée + variables d'environnement `IA_*` |

## Licence

Ce projet est fourni tel quel. Vérifiez les conditions de licence applicables avant toute utilisation.
