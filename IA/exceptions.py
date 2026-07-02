"""
IA/exceptions.py — Exceptions personnalisées pour le module IA.

Hiérarchie :
    IAError (base)
    ├── ModelNotFoundError
    ├── ModelFormatError
    ├── InferenceError
    ├── TrainingError
    ├── ConfigurationError
    └── DatasetError
"""

__all__ = [
    "IAError", "ModelNotFoundError", "ModelFormatError",
    "InferenceError", "TrainingError", "ConfigurationError",
    "DatasetError",
]


class IAError(Exception):
    """Exception de base pour tous les erreurs du module IA."""

    def __init__(self, message: str, *, model_type: str = None, details: str = None):
        self.model_type = model_type
        self.details = details
        full = message
        if model_type:
            full = f"[{model_type}] {message}"
        if details:
            full = f"{full} — {details}"
        super().__init__(full)


class ModelNotFoundError(IAError):
    """Le fichier modèle demandé n'existe pas ou est illisible."""
    pass


class ModelFormatError(IAError):
    """Le fichier modèle a un format invalide (magic, version, clés manquantes)."""
    pass


class InferenceError(IAError):
    """Erreur survenue pendant l'inférence (shape mismatch, clé manquante, etc.)."""
    pass


class TrainingError(IAError):
    """Erreur survenue pendant l'entraînement (divergence, données invalides)."""
    pass


class ConfigurationError(IAError):
    """Configuration invalide (paramètres manquants, valeurs hors bornes)."""
    pass


class DatasetError(IAError):
    """Erreur de chargement ou de format de dataset."""
    pass