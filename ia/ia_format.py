"""
IA/ia_format.py — Format binaire natif .ia pour la sérialisation des modèles.

Format :
  Offset  Taille  Champ
  0       4       Magique : "IAV3" (4 bytes)
  4       2       Version majeure (uint16 LE)
  6       2       Version mineure (uint16 LE)
  8       4       Taille du header JSON (uint32 LE)
  12      N       Header JSON (UTF-8) : type, config, accuracy, history...
  12+N    4       Nombre de tenseurs (uint32 LE)
  16+N    ...     Pour chaque tenseur :
                    - 1 byte : dtype (0=f32, 1=f64, 2=i64)
                    - 4 bytes : ndim (uint32 LE)
                    - 8*ndim bytes : shape (int64 LE)
                    - 4 bytes : taille du nom (uint32 LE)
                    - M bytes : nom (UTF-8)
                    - produit(shape) * sizeof(dtype) bytes : données brutes

Le format est volontairement simple et sans dépendance (juste struct + json).
Il peut être lu en C++ via memcpy/fread sans bibliothèque externe.
"""
import struct
import json
import numpy as np
from typing import Any, Dict, List, Tuple

MAGIC = b"IAV3"
VERSION_MAJOR = 1
VERSION_MINOR = 0

_DTYPE_MAP = {
    'float32': 0,
    'float64': 1,
    'int64':   2,
}
_DTYPE_REV = {v: k for k, v in _DTYPE_MAP.items()}
_DTYPE_SIZE = {0: 4, 1: 8, 2: 8}


def save_model(path: str, header: Dict[str, Any], tensors: Dict[str, np.ndarray]) -> None:
    """Sauvegarde un modèle au format .ia.

    Args:
        path: chemin du fichier (.ia recommandé).
        header: dictionnaire JSON-sérialisable contenant la config
                (type, hyperparams, accuracy, history...).
        tensors: dictionnaire {nom: ndarray} des poids du modèle.
    """
    header_bytes = json.dumps(header, ensure_ascii=False, default=_json_default).encode('utf-8')

    with open(path, 'wb') as f:
        # Magique + version
        f.write(MAGIC)
        f.write(struct.pack('<HH', VERSION_MAJOR, VERSION_MINOR))
        # Header
        f.write(struct.pack('<I', len(header_bytes)))
        f.write(header_bytes)
        # Tenseurs
        f.write(struct.pack('<I', len(tensors)))
        for name, arr in tensors.items():
            arr = np.asarray(arr)
            dtype_id = _DTYPE_MAP.get(arr.dtype.name)
            if dtype_id is None:
                # Conversion forcée vers float64 si dtype non supporté
                arr = arr.astype(np.float64)
                dtype_id = 1
            f.write(struct.pack('<B', dtype_id))
            f.write(struct.pack('<I', arr.ndim))
            for dim in arr.shape:
                f.write(struct.pack('<q', int(dim)))
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            # Données brutes en C-contiguous
            arr_c = np.ascontiguousarray(arr)
            f.write(arr_c.tobytes())


def load_model(path: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Charge un modèle .ia.

    Returns:
        (header, tensors) : header est un dict, tensors est {nom: ndarray}.
    """
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Fichier .ia invalide : magic={magic!r} (attendu {MAGIC!r})")
        vmaj, vmin = struct.unpack('<HH', f.read(4))
        if vmaj != VERSION_MAJOR:
            raise ValueError(f"Version .ia incompatible : {vmaj}.{vmin} (attendu {VERSION_MAJOR}.x)")
        header_len = struct.unpack('<I', f.read(4))[0]
        header = json.loads(f.read(header_len).decode('utf-8'))
        n_tensors = struct.unpack('<I', f.read(4))[0]
        tensors = {}
        for _ in range(n_tensors):
            dtype_id = struct.unpack('<B', f.read(1))[0]
            ndim = struct.unpack('<I', f.read(4))[0]
            shape = tuple(struct.unpack('<q', f.read(8))[0] for _ in range(ndim))
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            n_elem = 1
            for d in shape:
                n_elem *= d
            dtype = np.dtype(_DTYPE_REV[dtype_id])
            data = f.read(n_elem * _DTYPE_SIZE[dtype_id])
            arr = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
            tensors[name] = arr
    return header, tensors


def _json_default(o):
    """Conversion par défaut pour les types non-JSON."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Type non JSON-sérialisable : {type(o)}")


def model_info(path: str) -> Dict[str, Any]:
    """Retourne un résumé du modèle sans charger les tenseurs (léger)."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Fichier .ia invalide")
        vmaj, vmin = struct.unpack('<HH', f.read(4))
        header_len = struct.unpack('<I', f.read(4))[0]
        header = json.loads(f.read(header_len).decode('utf-8'))
    header['_format_version'] = f"{vmaj}.{vmin}"
    return header
