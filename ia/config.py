"""
IA/config.py — Configuration centralisée du module IA.

Toutes les constantes sont définies ici. Aucune valeur n'est hardcodée
dans les autres modules. Surchargez via variables d'environnement IA_*.
"""

import os

# ---------------------------------------------------------------------------
# Chemins de base
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(_BASE_DIR, "models")

# ---------------------------------------------------------------------------
# Entraînement global
# ---------------------------------------------------------------------------
TRAIN_LR = float(os.environ.get("IA_LR", "0.01"))
TRAIN_EPOCHS = int(os.environ.get("IA_EPOCHS", "1000"))
TRAIN_BATCH_SIZE = int(os.environ.get("IA_BATCH_SIZE", "32"))
TRAIN_SEED = int(os.environ.get("IA_SEED", "42"))
TRAIN_EARLY_STOP_LOSS = float(os.environ.get("IA_EARLY_STOP_LOSS", "0.001"))
TRAIN_GRADIENT_CLIP = float(os.environ.get("IA_GRADIENT_CLIP", "1.0"))
TRAIN_D_STEPS_PER_G = int(os.environ.get("IA_D_STEPS_PER_G", "2"))

# ---------------------------------------------------------------------------
# Modèle de sauvegarde (.gy)
# ---------------------------------------------------------------------------
MODEL_EXTENSION = ".gy"
MODEL_SAVE_BEST_ONLY = os.environ.get("IA_SAVE_BEST_ONLY", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_config(overrides: dict | None = None) -> dict:
    """Retourne un dictionnaire plat de toute la configuration."""
    import types
    _skip = {"os", "get_config", "ensure_directories", "types"}
    cfg = {}
    for name, val in globals().items():
        if name.startswith("_") or name in _skip:
            continue
        if not isinstance(val, types.ModuleType) and not callable(val):
            cfg[name] = val
    if overrides:
        cfg.update(overrides)
    return cfg


def ensure_directories():
    """Crée les répertoires requis s'ils n'existent pas."""
    os.makedirs(MODELS_DIR, exist_ok=True)