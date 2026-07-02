"""
IA/_utils.py — Utilitaires partagés pour les modules train et infer.

Ce module centralise les opérations répétées à travers les 8 modèles
pour éviter la duplication de code et assurer la cohérence.
"""

import logging
from typing import Any, Dict, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Reconstruction de listes indexées depuis les tensors sérialisés
# ============================================================================

def reconstruct_indexed_lists(
    tensors: Dict[str, Any],
    model: Dict[str, Any],
) -> None:
    """Reconstruit les listes indexées (k_0, k_1, ...) dans *model* à partir de *tensors*.

    Quand ``serialize_model_dict`` sérialise une liste de ndarrays, elle l'aplatit
    en clés indexées (``weights_0``, ``weights_1``, …).  Cette fonction les
    reconstruit en listes Python dans le dictionnaire *model*.

    Args:
        tensors: Dictionnaire {nom: ndarray} issu de ``ia_format.load_model``.
        model: Dictionnaire cible qui sera modifié en place.
    """
    list_bases: Set[str] = set()
    for k in tensors:
        if '_' in k:
            base, idx = k.rsplit('_', 1)
            if idx.isdigit():
                list_bases.add(base)

    for base in sorted(list_bases):
        indexed = sorted(
            [
                (int(k.rsplit('_', 1)[1]), tensors[k])
                for k in tensors
                if k.startswith(base + '_')
                and k[len(base) + 1:].isdigit()
            ],
            key=lambda x: x[0],
        )
        if indexed:
            model[base] = [v for _, v in indexed]


# ============================================================================
# Fusion des formats V2/V3 de header
# ============================================================================

def merge_header(header: Dict[str, Any], tensors: Dict[str, Any]) -> Dict[str, Any]:
    """Fusionne un header et ses tensors en un dictionnaire unifié.

    Gère à la fois le format V2 (config à plat dans header) et V3
    (config imbriqué sous ``header['config']``).

    Args:
        header: Dictionnaire JSON issu de ``ia_format.load_model``.
        tensors: Dictionnaire {nom: ndarray}.

    Returns:
        dict: Dictionnaire fusionné avec config, tensors, et listes reconstruites.
    """
    v3_config = header.get('config', {})
    model = dict(header)
    model.update(v3_config)
    model.pop('config', None)  # éviter la clé imbriquée redondante
    model.update(tensors)
    reconstruct_indexed_lists(tensors, model)
    return model


# ============================================================================
# Validation d'entrées
# ============================================================================

def validate_ndarray(
    data: Any,
    name: str = "input",
    expected_ndim: int = None,
    min_dim_size: int = 1,
) -> np.ndarray:
    """Valide et convertit une entrée en ndarray numpy.

    Args:
        data: Données d'entrée (ndarray, liste, ou scalaire).
        name: Nom descriptif pour les messages d'erreur.
        expected_ndim: Nombre de dimensions attendu (None = pas de vérification).
        min_dim_size: Taille minimale acceptée pour chaque dimension.

    Returns:
        np.ndarray: Données converties en float64.

    Raises:
        ValueError: Si les données sont invalides.
    """
    if data is None:
        raise ValueError(f"{name}: les données ne peuvent pas être None")
    arr = np.asarray(data, dtype=np.float64)
    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValueError(
            f"{name}: attendu ndim={expected_ndim}, obtenu ndim={arr.ndim} "
            f"(shape={arr.shape})"
        )
    if min_dim_size > 1:
        for i, d in enumerate(arr.shape):
            if d < min_dim_size:
                raise ValueError(
                    f"{name}: la dimension {i} a une taille {d} < {min_dim_size}"
                )
    return arr


def validate_model_path(path: str) -> str:
    """Valide qu'un chemin de modèle existe et est lisible.

    Args:
        path: Chemin vers le fichier .gy.

    Returns:
        str: Le chemin absolu validé.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    import os
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return os.path.abspath(path)


def validate_sequence(sequence: Any, name: str = "sequence") -> np.ndarray:
    """Valide une séquence d'entrée (RNN, Transformer, SLM).

    Args:
        sequence: Données de séquence.
        name: Nom descriptif pour les messages d'erreur.

    Returns:
        np.ndarray: Séquence validée comme float64.
    """
    if sequence is None:
        raise ValueError(f"{name}: la séquence ne peut pas être None")
    arr = np.asarray(sequence, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"{name}: attendu au moins 2D, obtenu {arr.ndim}D")
    return arr