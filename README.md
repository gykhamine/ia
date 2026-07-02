# IA — Module d'Intelligence Artificielle

Moteur de deep learning en C++ avec bindings Python. Architecture modulaire pour l'entraînement et l'inférence de modèles deep learning avec un moteur de calcul C++ pur (ctypes) remplaçant numpy/pandas pour toutes les opérations mathématiques fondamentales.

## Licence

GNU General Public License v3.0 (GPL-3.0)

## Déploiement GitHub

```bash
# Cloner le dépôt
git clone https://github.com/<user>/IA.git
cd IA

# Le module IA est prêt à l'import
# Le .so compilé est dans IA/cpp/_ia_core.cpython-314-x86_64-linux-gnu.so
# Si vous changez d'environnement, recompilez :
cd IA/cpp && make
```

### Prérequis serveur de production

- **OS** : Linux x86_64
- **Python** : 3.14 (même version que le `.so` compilé)
- **Dépendances** : `numpy`
- Si le serveur a une version différente de Python, recompilez `_ia_core.so` via le `Makefile`

### Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `IA_LR` | `0.01` | Taux d'apprentissage global |
| `IA_EPOCHS` | `1000` | Nombre d'époques global |
| `IA_BATCH_SIZE` | `32` | Taille de batch global |
| `IA_SEED` | `42` | Graine aléatoire |
| `IA_EARLY_STOP_LOSS` | `0.001` | Seuil d'arrêt précoce |
| `IA_GRADIENT_CLIP` | `1.0` | Clip des gradients |
| `IA_D_STEPS_PER_G` | `2` | Pas de D par pas de G (GAN) |
| `IA_SAVE_BEST_ONLY` | `true` | Sauvegarder uniquement le meilleur modèle |

---

## Structure du module

```
IA/
├── __init__.py          # Point d'entrée, raccourcis V2
├── config.py            # Configuration centralisée
├── dataset.py           # Chargement multi-formats
├── trainer.py           # API unifiée V3 (Trainer + Model)
├── model.py             # Classe Model (wrapper inférence + persistance)
├── callbacks.py         # Callbacks Keras-like
├── ia_format.py         # Format binaire natif .gy
├── exceptions.py        # Exceptions personnalisées
├── _utils.py            # Utilitaires internes
├── cpp/                 # Moteur de calcul C++
│   ├── _ia_core.so      # Extension compilée
│   ├── engine.h/cpp     # Moteur tensoriel
│   ├── autograd.h/cpp   # Différentiation automatique
│   └── c_api.cpp        # API C pour ctypes
├── train/               # Entraînement de tous les modèles
│   ├── mlp.py
│   ├── rnn.py
│   ├── cnn.py
│   ├── transformer.py
│   ├── gan.py
│   ├── ldm.py
│   ├── slm.py
│   └── vision.py
├── infer/               # Inférence de tous les modèles
│   ├── mlp.py
│   ├── rnn.py
│   ├── cnn.py
│   ├── transformer.py
│   ├── gan.py
│   ├── ldm.py
│   ├── slm.py
│   └── vision.py
└── models/              # Répertoire de sauvegarde par défaut
```

---

## API complète — Signatures exhaustives

### 1. MLP (Multi-Layer Perceptron)

Classification et régression tabulaire. Binaire ou multi-classes automatique.

#### Entraînement

```python
IA.train_mlp(
    X: np.ndarray = None,       # (n_samples, n_features)
    y: np.ndarray = None,       # (n_samples, 1)
    hidden_sizes: list = None,  # Tailles des couches cachées, défaut [64, 32]
    lr: float = 0.01,
    epochs: int = 500,
    save_path: str = None,      # Chemin .gy, défaut models/mini_mlp.gy
    seed: int = 42,
    batch_size: int = 256,
    dropout: float = 0.0,
) -> dict  # {'model', 'save_path', 'accuracy', 'history'}
```

#### Inférence

```python
model = IA.load_mlp(path: str) -> dict
result = IA.predict_mlp(
    model: dict,
    X: np.ndarray,              # (n_samples, n_features) ou (n_features,)
) -> dict  # {'output', 'predictions', 'probabilities'} (multiclass) ou {'output', 'predictions'} (binaire)
```

#### Django — Vue

```python
# views.py
import numpy as np
from django.http import JsonResponse
import IA

MODEL_PATH = "models/mlp.gy"
_mlp_model = None

def get_mlp_model():
    global _mlp_model
    if _mlp_model is None:
        _mlp_model = IA.load_mlp(MODEL_PATH)
    return _mlp_model

def predict_mlp_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST requis"}, status=405)
    data = np.array(request.POST.getlist("features[]"), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    model = get_mlp_model()
    result = IA.predict_mlp(model, data)
    return JsonResponse({
        "predictions": result["predictions"].tolist(),
        "output": result["output"].tolist(),
    })
```

#### Django — Tâche Celery

```python
# tasks.py
from celery import shared_task
import IA
import numpy as np

@shared_task
def train_mlp_task(features, labels, epochs=3000):
    X = np.array(features, dtype=np.float64)
    y = np.array(labels, dtype=np.float64).reshape(-1, 1)
    result = IA.train_mlp(X=X, y=y, hidden_sizes=[64, 32], epochs=epochs,
                          save_path="models/mlp.gy")
    return {"accuracy": result["accuracy"], "save_path": result["save_path"]}
```

---

### 2. RNN (Réseau Récurrent)

Classification de séquences. Supporte l'empilement de couches (stacked RNN).

#### Entraînement

```python
IA.train_rnn(
    X: np.ndarray = None,       # (batch, seq_len, input_size)
    y: np.ndarray = None,       # (batch, 1)
    hidden_size: int = 4,
    lr: float = 0.1,
    epochs: int = 1000,
    save_path: str = None,
    seed: int = 42,
    grad_clip: float = 1.0,
    num_layers: int = 1,        # Nombre de couches RNN empilées
) -> dict  # {'model', 'save_path', 'accuracy', 'history'}
```

#### Inférence

```python
model = IA.load_rnn(path: str) -> dict
result = IA.predict_rnn(
    model: dict,
    sequence: np.ndarray,       # (seq_len, input_size)
) -> dict  # {'probability', 'class', 'confidence', 'hidden_states'}
```

#### Django

```python
# views.py
def predict_rnn_view(request):
    seq = np.array(request.POST.getlist("sequence[]"), dtype=np.float64)
    model = IA.load_rnn("models/rnn.gy")
    result = IA.predict_rnn(model, seq)
    return JsonResponse(result)
```

---

### 3. CNN 2D

Classification de motifs 2D (images grayscale).

#### Entraînement

```python
IA.train_cnn2d(
    X: np.ndarray = None,       # (n_samples, H, W)
    y: np.ndarray = None,       # (n_samples, 1)
    input_shape: tuple = (5, 5),
    kernel_shape: tuple = (3, 3),
    lr: float = 0.01,
    epochs: int = 1000,
    early_stop_loss: float = 0.001,
    save_path: str = None,
    seed: int = 42,
    num_conv_layers: int = 1,   # Couches de convolution empilées
) -> dict  # {'model', 'save_path', 'accuracy', 'final_loss', 'history'}
```

#### Inférence

```python
model = IA.load_cnn2d(path: str) -> dict
result = IA.predict_cnn2d(
    model: dict,
    image: np.ndarray,          # (H, W)
) -> dict  # {'probability', 'class', 'confidence', 'activations'}
# class = 'STRUCTURÉ' ou 'ALÉATOIRE'
```

#### Django

```python
# views.py
def predict_cnn2d_view(request):
    image = np.array(request.POST.getlist("image[]"), dtype=np.float64)
    H, W = int(request.POST.get("height")), int(request.POST.get("width"))
    image = image.reshape(H, W)
    model = IA.load_cnn2d("models/cnn2d.gy")
    result = IA.predict_cnn2d(model, image)
    return JsonResponse(result)
```

---

### 4. CNN N-D

Classification de volumes N-dimensionnels (3D, 4D, 5D...).

#### Entraînement

```python
IA.train_cnn_nd(
    dimensions: int = 4,
    volume_shape: tuple = None,  # Défaut (3,3,3,3)[:dimensions]
    kernel_shape: tuple = None,  # Défaut (2,2,2,2)[:dimensions]
    lr: float = 0.01,
    epochs: int = 1000,
    save_path: str = None,
    seed: int = 42,
    num_conv_layers: int = 1,
) -> dict  # {'model', 'save_path', 'accuracy', 'history'}
```

#### Inférence

```python
model = IA.load_cnn_nd(path: str) -> dict
result = IA.predict_cnn_nd(
    model: dict,
    volume: np.ndarray,         # N-D array
) -> dict  # {'probability', 'class', 'confidence', 'activations'}
```

---

### 5. Transformer (single-head)

Classification de séquences de tokens. Supporte multi-blocs.

#### Entraînement

```python
IA.train_transformer(
    X: np.ndarray = None,       # (batch, seq_len) — tokens entiers < vocab_size
    y: np.ndarray = None,       # (batch, 1)
    seq_len: int = 4,
    vocab_size: int = 6,
    embed_dim: int = 8,
    ff_dim: int = 16,
    lr: float = 0.01,
    epochs: int = 2000,
    early_stop_loss: float = 0.001,
    save_path: str = None,
    seed: int = 42,
    num_blocks: int = 1,        # Nombre de blocs transformer
) -> dict  # {'model', 'save_path', 'accuracy', 'history'}
```

#### Inférence

```python
model = IA.load_transformer(path: str) -> dict
result = IA.predict_transformer(
    model: dict,
    tokens: list[int],          # Liste d'entiers
) -> dict  # {'probability', 'class', 'confidence', 'attention_weights'}
```

#### Django

```python
# views.py
def predict_transformer_view(request):
    tokens = [int(t) for t in request.POST.getlist("tokens[]")]
    model = IA.load_transformer("models/transformer.gy")
    result = IA.predict_transformer(model, tokens)
    return JsonResponse(result)
```

---

### 6. Transformer 3D (multi-head)

Classification de séquences avec attention multi-têtes et batch.

#### Entraînement

```python
IA.train_transformer3d(
    X: np.ndarray = None,       # (batch, seq_len) — tokens entiers
    y: np.ndarray = None,       # (batch, 1)
    vocab_size: int = 6,
    seq_len: int = 4,
    embed_dim: int = 8,
    ff_dim: int = 16,
    num_heads: int = 2,
    lr: float = 0.01,
    epochs: int = 5000,
    convergence_loss: float = 0.01,
    save_path: str = None,
    seed: int = 42,
) -> dict  # {'model', 'save_path', 'accuracy', 'history'}
```

#### Inférence

```python
model = IA.load_transformer3d(path: str)  # -> MiniTransformer3D instance
result = IA.predict_transformer3d(
    model,                        # Instance MiniTransformer3D
    tokens: list[int],
) -> dict  # {'probability', 'class', 'confidence', 'attention_weights'}
```

---

### 7. GAN 1D

Génération de valeurs scalaires (mélange de 2 gaussiennes).

#### Entraînement

```python
IA.train_gan_1d(
    latent_dim: int = 2,
    hidden_dim: int = 16,
    lr: float = 0.001,
    epochs: int = 5000,
    batch_size: int = 32,
    save_path: str = None,
    seed: int = 42,
    generator_layers: list = None,       # [latent_dim, h, h, 1]
    discriminator_layers: list = None,   # [1, h, h, 1]
) -> dict  # {'model', 'save_path', 'accuracy', 'history'}
```

#### Inférence

```python
model = IA.load_gan_1d(path: str) -> dict
result = IA.generate_gan_1d(
    model: dict,
    z: np.ndarray = None,      # Vecteur latent, None = aléatoire
    num_samples: int = 1,
) -> dict  # {'generated': ndarray, 'latent_vectors': ndarray}
```

#### Django

```python
# views.py
def generate_gan_view(request):
    model = IA.load_gan_1d("models/gan_1d.gy")
    n = int(request.GET.get("n", 10))
    result = IA.generate_gan_1d(model, num_samples=n)
    return JsonResponse({"generated": result["generated"].tolist()})
```

---

### 8. GAN N-D

Génération de vecteurs de dimension `data_dim`. Sortie tanh.

#### Entraînement

```python
IA.train_gan_nd(
    data_dim: int = 16,
    latent_dim: int = 16,
    hidden_dim: int = 64,
    lr: float = 0.001,
    epochs: int = 5000,
    batch_size: int = 32,
    save_path: str = None,
    seed: int = 42,
    generator_layers: list = None,
    discriminator_layers: list = None,
) -> dict
```

#### Inférence

```python
model = IA.load_gan_nd(path: str) -> dict
result = IA.generate_gan_nd(
    model: dict,
    z: np.ndarray = None,
    num_samples: int = 1,
) -> dict  # {'generated', 'latent_vectors'}
```

---

### 9. GAN 3D

Génération de volumes 3D (4x4x4 = 64 voxels par défaut). Sortie tanh.

#### Entraînement

```python
IA.train_gan_3d(
    volume_size: int = 4,
    latent_dim: int = 16,
    hidden_dim: int = 128,
    lr: float = 0.0005,
    epochs: int = 5000,
    batch_size: int = 32,
    save_path: str = None,
    seed: int = 42,
    generator_layers: list = None,
    discriminator_layers: list = None,
) -> dict
```

#### Inférence

```python
model = IA.load_gan_3d(path: str) -> dict
result = IA.generate_gan_3d(
    model: dict,
    z: np.ndarray = None,      # (1, latent_dim)
) -> dict  # {'generated_volume': ndarray, 'volume_shape': tuple}
```

---

### 10. GAN RGB

Génération d'images RGB 32x32x3 (3072 pixels). Sortie tanh.

#### Entraînement

```python
IA.train_gan_rgb(
    image_size: int = 32,
    channels: int = 3,
    latent_dim: int = 200,
    hidden_dim: int = 512,
    lr: float = 0.00002,
    epochs: int = 10,
    batch_size: int = 16,
    save_path: str = None,
    seed: int = 42,
    generator_layers: list = None,
    discriminator_layers: list = None,
) -> dict
```

#### Inférence

```python
model = IA.load_gan_rgb(path: str) -> dict
result = IA.generate_gan_rgb(
    model: dict,
    z: np.ndarray = None,
    num_samples: int = 1,
) -> dict  # {'generated_images': ndarray (N, H, W, 3), 'image_shape': tuple}
```

---

### 11. LDM Image

Diffusion conditionnelle pour la génération d'images 2D. Débruitage DDPM.

#### Entraînement

```python
IA.train_ldm_image(
    image_size: int = 8,
    num_classes: int = 5,
    timesteps: int = 200,
    lr: float = 0.001,
    epochs: int = 1000,
    save_path: str = None,
    seed: int = 42,
    hidden_sizes: list = None,   # Profondeur du réseau, None = 2 couches
) -> dict  # {'model', 'save_path', 'accuracy', 'history'}
```

#### Inférence

```python
model = IA.load_ldm_image(path: str) -> dict
result = IA.generate_ldm_image(
    model: dict,
    class_id: int,              # Classe conditionnelle
    shape: tuple = (8, 8),      # Forme de l'image
    num_steps: int = 50,        # Étapes de débruitage
) -> dict  # {'generated': ndarray (H, W), 'class_id': int, 'num_steps': int}
```

#### Django

```python
# views.py
def generate_ldm_image_view(request):
    model = IA.load_ldm_image("models/ldm_image.gy")
    class_id = int(request.GET.get("class_id", 0))
    result = IA.generate_ldm_image(model, class_id=class_id, shape=(8, 8))
    return JsonResponse({"image": result["generated"].tolist()})
```

---

### 12. LDM Audio

Diffusion conditionnelle pour la génération de signaux audio.

#### Entraînement

```python
IA.train_ldm_audio(
    signal_length: int = 64,
    num_classes: int = 5,
    timesteps: int = 200,
    lr: float = 0.001,
    epochs: int = 1000,
    save_path: str = None,
    seed: int = 42,
    hidden_sizes: list = None,
) -> dict  # {'model', 'save_path', 'accuracy', 'history'}
```

#### Inférence

```python
model = IA.load_ldm_audio(path: str) -> dict
result = IA.generate_ldm_audio(
    model: dict,
    class_id: int,
    signal_length: int = 64,
    num_steps: int = 50,
) -> dict  # {'generated': ndarray (signal_length,), 'class_id': int, 'num_steps': int}
```

---

### 13. SLM — Détection d'émotion

6 classes : joie, tristesse, colère, peur, surprise, neutre.

#### Entraînement

```python
IA.train_slm_emotion(
    sentences: list[str] = None,  # Phrases françaises
    labels: list[int] = None,     # Indices 0-5
    seq_len: int = 10,
    embed_dim: int = 16,
    ff_dim: int = 32,
    lr: float = 0.01,
    epochs: int = 500,
    save_path: str = None,
    seed: int = 42,
    num_blocks: int = 1,
) -> dict  # {'model', 'save_path', 'accuracy', 'history'}
```

#### Inférence

```python
model = IA.load_slm(path: str) -> dict
result = IA.predict_slm(
    model: dict,
    text: str,
) -> dict  # {'predictions': [{'class': str, 'probability': float}, ...], 'attention_weights'}
```

#### Django

```python
# views.py
def predict_emotion_view(request):
    text = request.POST.get("text", "")
    model = IA.load_slm("models/slm_emotion.gy")
    result = IA.predict_slm(model, text)
    return JsonResponse({
        "emotion": result["predictions"][0]["class"],
        "confidence": result["predictions"][0]["probability"],
        "all_predictions": result["predictions"],
    })
```

---

### 14. SLM — Détection d'humeur

8 classes : joyeux, triste, énergique, calme, stressé, fatigué, motivé, anxieux.

#### Entraînement

```python
IA.train_slm_mood(
    sentences: list[str] = None,
    labels: list[int] = None,     # Indices 0-7
    seq_len: int = 10,
    embed_dim: int = 16,
    ff_dim: int = 32,
    lr: float = 0.01,
    epochs: int = 500,
    save_path: str = None,
    seed: int = 42,
    num_blocks: int = 1,
) -> dict
```

Même API d'inférence que `predict_slm`.

---

### 15. SLM — Type de phrase

5 classes : question, affirmation, ordre, conseil, exclamation.

#### Entraînement

```python
IA.train_slm_statement(
    sentences: list[str] = None,
    labels: list[int] = None,     # Indices 0-4
    seq_len: int = 10,
    embed_dim: int = 16,
    ff_dim: int = 32,
    lr: float = 0.01,
    epochs: int = 500,
    save_path: str = None,
    seed: int = 42,
    num_blocks: int = 1,
) -> dict
```

Même API d'inférence que `predict_slm`.

---

### 16. SLM — Analyse de sentiment

3 classes : positif, négatif, neutre.

#### Entraînement

```python
IA.train_slm_sentiment(
    sentences: list[str] = None,
    labels: list[int] = None,     # Indices 0-2
    seq_len: int = 10,
    embed_dim: int = 16,
    ff_dim: int = 32,
    lr: float = 0.01,
    epochs: int = 500,
    save_path: str = None,
    seed: int = 42,
    num_blocks: int = 1,
) -> dict
```

Même API d'inférence que `predict_slm`.

---

### 17. SLM — Prédiction du mot suivant

Prédit les mots les plus probables suite à un contexte.

#### Entraînement

```python
IA.train_slm_next_word(
    vocab: dict = None,           # mot -> index (défaut : 10 mots français)
    seq_len: int = 4,
    embed_dim: int = 16,
    ff_dim: int = 32,
    lr: float = 0.01,
    epochs: int = 1000,
    save_path: str = None,
    seed: int = 42,
    num_blocks: int = 1,
) -> dict
```

#### Inférence

```python
model = IA.load_slm(path: str) -> dict
result = IA.predict_slm_next_word(
    model: dict,
    context_tokens: list[int],
) -> dict  # {'top_words': [{'word': str, 'probability': float}, ...]}
```

#### Django

```python
# views.py
def next_word_view(request):
    tokens = [int(t) for t in request.POST.getlist("tokens[]")]
    model = IA.load_slm("models/slm_next_word.gy")
    result = IA.predict_slm_next_word(model, tokens)
    return JsonResponse(result)
```

---

### 18. Image Classifier

Classification d'images à partir de features extraites (histogramme RGB, gradient, LBP, moments).

#### Entraînement

```python
IA.train_image_classifier(
    X: np.ndarray = None,       # (N, feature_dim)
    y: np.ndarray = None,       # (N,) — labels entiers
    num_classes: int = 3,
    feature_dim: int = 128,
    hidden_dim: int = 64,
    lr: float = 0.01,
    epochs: int = 500,
    save_path: str = None,
    seed: int = 42,
    hidden_layers: list = None,  # Liste de tailles, None = 1 couche cachée
) -> dict
```

#### Inférence

```python
model = IA.load_image_classifier(path: str) -> dict
result = IA.predict_image(
    model: dict,
    image: np.ndarray,          # (H, W) ou (H, W, 3)
) -> dict  # {'class': str, 'confidence': float, 'probabilities': dict, 'feature_vector': ndarray}
```

#### Django

```python
# views.py
def classify_image_view(request):
    import json
    image_data = json.loads(request.POST.get("image"))
    image = np.array(image_data, dtype=np.float64)
    model = IA.load_image_classifier("models/image_clf.gy")
    result = IA.predict_image(model, image)
    return JsonResponse(result)
```

---

### 19. Speech Classifier

Classification de phonèmes à partir de features audio (ZCR, énergie, spectre, etc.).

#### Entraînement

```python
IA.train_speech_classifier(
    X: np.ndarray = None,       # (N, feature_dim)
    y: np.ndarray = None,       # (N,) — labels entiers
    num_classes: int = 3,
    feature_dim: int = 26,
    hidden_dim: int = 64,
    lr: float = 0.01,
    epochs: int = 500,
    save_path: str = None,
    seed: int = 42,
    hidden_layers: list = None,
) -> dict
```

#### Inférence

```python
model = IA.load_speech_classifier(path: str) -> dict
result = IA.predict_speech(
    model: dict,
    audio_signal: np.ndarray,   # 1D signal
) -> dict  # {'class': str, 'confidence': float, 'probabilities': dict, 'feature_vector': ndarray}
```

---

## Fonctions utilitaires

### Dataset

```python
# Charger des données multi-formats (CSV, JSON, NPY, Parquet, HDF5, XLSX, Pickle...)
X, y = IA.load_dataset(path: str, target: str = None)
# target = nom de la colonne label. Si None, retourne (data, None).

# Formats supportés
IA.supported_formats  # -> ['.csv', '.tsv', '.json', '.jsonl', '.npy', '.npz', '.txt', '.parquet', '.h5', '.hdf5', '.xlsx', '.xls', '.pkl', '.pickle']

# Infos sur un fichier
IA.dataset_info(path: str) -> dict
```

### Configuration

```python
# Récupérer toute la configuration
IA.get_config(overrides: dict = None) -> dict

# Créer les répertoires requis
IA.ensure_directories()
```

---

## API unifiée V3 (Trainer + Model)

```python
from IA import Trainer

# Entraînement unifié
trainer = Trainer(verbose=True)
model = trainer.train(type='rnn', epochs=500, lr=0.01)
y = model.predict(X)
model.save('mon_modele.gy')

# Chargement
model2 = Trainer.load('mon_modele.gy')
y_pred = model2.predict(X_new)
```

Types supportés par `Trainer.train(type=...)` :
`mlp`, `rnn`, `cnn2d`, `cnn_nd`, `transformer`, `transformer3d`, `gan_1d`, `gan_nd`, `gan_3d`, `gan_rgb`, `ldm_image`, `ldm_audio`, `slm_next_word`, `slm_emotion`, `slm_mood`, `slm_statement`, `slm_sentiment`, `image_classifier`, `speech_classifier`

### Callbacks

```python
from IA import EarlyStopping, ModelCheckpoint, CSVLogger, ProgressPrinter

callbacks = [
    EarlyStopping(monitor='loss', patience=10, min_delta=0.001),
    ModelCheckpoint(filepath='best_model.gy', save_best_only=True),
    CSVLogger('training_log.csv'),
    ProgressPrinter(),
]

trainer = Trainer(callbacks=callbacks)
model = trainer.train(type='mlp', epochs=5000)
```

---

## Exceptions

```python
from IA import IAError, ModelNotFoundError, ModelFormatError
from IA import InferenceError, TrainingError, ConfigurationError, DatasetError
```

---

## Intégration Django complète

### Structure recommandée

```
mon_erp/
├── ia_app/
│   ├── views.py
│   ├── tasks.py          # Tâches Celery
│   ├── urls.py
│   └── ia_config.py      # Configuration IA
├── models/               # Modèles .gy entraînés
│   ├── mlp.gy
│   ├── slm_emotion.gy
│   └── ...
├── celery.py
└── requirements.txt
```

### Configuration Django (`ia_app/ia_config.py`)

```python
import os
import IA

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
IA.ensure_directories()

# Cache des modèles chargés (singleton par processus)
_model_cache = {}

def get_model(name, load_fn, path):
    if name not in _model_cache:
        _model_cache[name] = load_fn(path)
    return _model_cache[name]
```

### URLs (`ia_app/urls.py`)

```python
from django.urls import path
from . import views

urlpatterns = [
    path('predict/mlp/', views.predict_mlp_view, name='predict_mlp'),
    path('predict/rnn/', views.predict_rnn_view, name='predict_rnn'),
    path('predict/cnn2d/', views.predict_cnn2d_view, name='predict_cnn2d'),
    path('predict/transformer/', views.predict_transformer_view, name='predict_transformer'),
    path('predict/emotion/', views.predict_emotion_view, name='predict_emotion'),
    path('predict/sentiment/', views.predict_sentiment_view, name='predict_sentiment'),
    path('predict/image/', views.classify_image_view, name='classify_image'),
    path('predict/speech/', views.predict_speech_view, name='predict_speech'),
    path('generate/gan1d/', views.generate_gan_view, name='generate_gan1d'),
    path('generate/ldm-image/', views.generate_ldm_image_view, name='generate_ldm_image'),
    path('generate/ldm-audio/', views.generate_ldm_audio_view, name='generate_ldm_audio'),
    path('generate/next-word/', views.next_word_view, name='next_word'),
    path('train/start/', views.start_training, name='start_training'),
    path('train/status/<task_id>/', views.training_status, name='training_status'),
]
```

### Tâches Celery (`ia_app/tasks.py`)

```python
from celery import shared_task
import numpy as np
import IA

@shared_task(bind=True)
def train_task(self, model_type, params):
    """Dispatch vers la bonne fonction d'entraînement."""
    train_fn = getattr(IA, f'train_{model_type}', None)
    if train_fn is None:
        return {"error": f"Type {model_type} non supporté"}

    result = train_fn(**params)
    self.update_state(state='SUCCESS', meta=result)
    return result

@shared_task
def train_slm_emotion_task(sentences, labels, epochs=500):
    result = IA.train_slm_emotion(
        sentences=sentences,
        labels=labels,
        epochs=epochs,
        save_path='models/slm_emotion.gy',
    )
    return {"accuracy": result["accuracy"]}

@shared_task
def train_mlp_task(features, labels, epochs=3000):
    X = np.array(features, dtype=np.float64)
    y = np.array(labels, dtype=np.float64).reshape(-1, 1)
    result = IA.train_mlp(
        X=X, y=y, hidden_sizes=[64, 32],
        epochs=epochs, save_path='models/mlp.gy',
    )
    return {"accuracy": result["accuracy"]}
```

### Vue de lancement d'entraînement (`ia_app/views.py`)

```python
from django.http import JsonResponse
from .tasks import train_task, train_slm_emotion_task, train_mlp_task

def start_training(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST requis"}, status=405)

    import json
    body = json.loads(request.body)
    model_type = body.get("type")
    params = body.get("params", {})

    # Lancer en arrière-plan
    task = train_task.delay(model_type, params)
    return JsonResponse({"task_id": task.id, "status": "started"})

def training_status(request, task_id):
    from celery.result import AsyncResult
    task = AsyncResult(task_id)

    if task.state == "PENDING":
        return JsonResponse({"state": "PENDING"})
    elif task.state == "SUCCESS":
        return JsonResponse({"state": "SUCCESS", "result": task.result})
    elif task.state == "FAILURE":
        return JsonResponse({"state": "FAILURE", "error": str(task.info)})
    return JsonResponse({"state": task.state, "meta": task.info})
```

### Celery configuration (`celery.py`)

```python
from celery import Celery

app = Celery('mon_erp')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
```

### `requirements.txt`

```
numpy>=1.24
celery>=5.3
redis>=4.5
Django>=4.2
```

---

## Résumé exhaustif — Toutes les fonctions

### Entraînement (19 fonctions)

| Fonction | X requis ? | Description |
|----------|-----------|-------------|
| `train_mlp(X, y, ...)` | Oui | MLP binaire/multi-classes |
| `train_rnn(X, y, ...)` | Oui | RNN séquentiel |
| `train_cnn2d(X, y, ...)` | Oui | CNN 2D motifs |
| `train_cnn_nd(...)` | Non (auto) | CNN N-D volumes |
| `train_transformer(X, y, ...)` | Oui | Transformer single-head |
| `train_transformer3d(X, y, ...)` | Oui | Transformer multi-head |
| `train_gan_1d(...)` | Non (auto) | GAN scalaire |
| `train_gan_nd(...)` | Non (auto) | GAN vecteurs |
| `train_gan_3d(...)` | Non (auto) | GAN volumes 3D |
| `train_gan_rgb(...)` | Non (auto) | GAN images RGB |
| `train_ldm_image(...)` | Non (auto) | Diffusion images |
| `train_ldm_audio(...)` | Non (auto) | Diffusion audio |
| `train_slm_next_word(...)` | Non (vocab auto) | Mot suivant |
| `train_slm_emotion(sentences, labels, ...)` | Oui | 6 émotions |
| `train_slm_mood(sentences, labels, ...)` | Oui | 8 humeurs |
| `train_slm_statement(sentences, labels, ...)` | Oui | 5 types de phrases |
| `train_slm_sentiment(sentences, labels, ...)` | Oui | 3 sentiments |
| `train_image_classifier(X, y, ...)` | Oui | Classification images |
| `train_speech_classifier(X, y, ...)` | Oui | Classification audio |

### Inférence (30 fonctions)

| Fonction | Entrée | Sortie |
|----------|--------|--------|
| `load_mlp(path)` | chemin .gy | dict |
| `predict_mlp(model, X)` | (n, features) | {output, predictions} |
| `load_rnn(path)` | chemin .gy | dict |
| `predict_rnn(model, sequence)` | (seq_len, in) | {probability, class} |
| `load_cnn2d(path)` | chemin .gy | dict |
| `predict_cnn2d(model, image)` | (H, W) | {probability, class} |
| `load_cnn_nd(path)` | chemin .gy | dict |
| `predict_cnn_nd(model, volume)` | N-D array | {probability, class} |
| `load_transformer(path)` | chemin .gy | dict |
| `predict_transformer(model, tokens)` | list[int] | {probability, class} |
| `load_transformer3d(path)` | chemin .gy | MiniTransformer3D |
| `predict_transformer3d(model, tokens)` | list[int] | {probability, class} |
| `load_gan_1d(path)` | chemin .gy | dict |
| `generate_gan_1d(model, z, n)` | latent/None | {generated} |
| `load_gan_nd(path)` | chemin .gy | dict |
| `generate_gan_nd(model, z, n)` | latent/None | {generated} |
| `load_gan_3d(path)` | chemin .gy | dict |
| `generate_gan_3d(model, z)` | latent/None | {generated_volume} |
| `load_gan_rgb(path)` | chemin .gy | dict |
| `generate_gan_rgb(model, z, n)` | latent/None | {generated_images} |
| `load_ldm_image(path)` | chemin .gy | dict |
| `generate_ldm_image(model, cls, shape, steps)` | class_id | {generated} |
| `load_ldm_audio(path)` | chemin .gy | dict |
| `generate_ldm_audio(model, cls, len, steps)` | class_id | {generated} |
| `load_slm(path)` | chemin .gy | dict |
| `predict_slm(model, text)` | str | {predictions} |
| `predict_slm_next_word(model, tokens)` | list[int] | {top_words} |
| `load_image_classifier(path)` | chemin .gy | dict |
| `predict_image(model, image)` | (H, W) ou (H, W, 3) | {class, confidence} |
| `load_speech_classifier(path)` | chemin .gy | dict |
| `predict_speech(model, signal)` | 1D array | {class, confidence} |