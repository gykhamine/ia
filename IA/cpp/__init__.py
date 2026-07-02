"""
IA/cpp/__init__.py — Wrapper ctypes pour le moteur C++ (ZÉRO dépendance externe).

Charge _ia_core.so via ctypes (pas pybind11, pas CMake).
Le .so est compilé avec: cd IA && make
"""

import ctypes
import os
import sys
import logging
import numpy as np
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

_lib = None
_BACKEND = "none"

# ---------------------------------------------------------------------------
# Types ctypes
# ---------------------------------------------------------------------------
_c_double_p = ctypes.POINTER(ctypes.c_double)
_c_int64_p = ctypes.POINTER(ctypes.c_int64)
_c_int = ctypes.c_int
_c_int64 = ctypes.c_int64
_c_uint64 = ctypes.c_uint64
_c_double = ctypes.c_double
_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_int64_pp = ctypes.POINTER(ctypes.c_int64)


def _find_lib():
    """Cherche _ia_core*.so dans cpp/ (courant puis package)."""
    import glob as _glob
    # 1) cpp/ sous le répertoire courant
    for search_dir in (os.path.join(os.getcwd(), "cpp"),
                       os.path.dirname(os.path.abspath(__file__))):
        matches = _glob.glob(os.path.join(search_dir, "_ia_core*.so"))
        if matches:
            return matches[0]
    return None


def _load():
    global _lib, _BACKEND
    lib_path = _find_lib()
    if lib_path is None:
        logger.warning(
            "IA: _ia_core.so non trouve. Compilez avec: cd IA && make")
        _BACKEND = "fallback"
        return False
    try:
        _lib = ctypes.CDLL(lib_path)
        _BACKEND = "C++"
        logger.info("IA: _ia_core.so charge: %s", lib_path)
        return True
    except OSError as e:
        logger.warning("IA: echec chargement _ia_core.so: %s", e)
        _BACKEND = "fallback"
        return False


_loaded = _load()

# ---------------------------------------------------------------------------
# Helpers de conversion numpy <-> ctypes
# ---------------------------------------------------------------------------

_MAX_NDIM = 8


def _arr_info(arr):
    """numpy array -> (ctypes data ptr, ctypes shape ptr, int ndim)."""
    a = np.ascontiguousarray(arr, dtype=np.float64)
    data = a.ctypes.data_as(_c_double_p)
    shape = (ctypes.c_int64 * a.ndim)(*a.shape)
    return data, shape, a.ndim, a


def _make_result(data_ptr, out_shape, out_ndim):
    """Pointeur C + shape C -> numpy array + libere la memoire C."""
    ndim = out_ndim.value if isinstance(out_ndim, ctypes._SimpleCData) else out_ndim
    if not data_ptr:
        # ndim == -1 signale une exception C++ interceptee (voir cpp_get_last_error).
        if ndim == -1 and _lib is not None and hasattr(_lib, "cpp_get_last_error"):
            _lib.cpp_get_last_error.restype = ctypes.c_char_p
            msg = _lib.cpp_get_last_error()
            msg = msg.decode("utf-8", "replace") if msg else "erreur inconnue"
            raise RuntimeError(f"IA C++ engine: {msg}")
        raise RuntimeError("C++ function returned NULL")
    dims = tuple(out_shape[i] for i in range(ndim))
    n = 1
    for d in dims:
        n *= d
    arr = np.empty(n, dtype=np.float64)
    ctypes.memmove(arr.ctypes.data, data_ptr, n * 8)
    arr = arr.reshape(dims)
    _lib.cpp_free(data_ptr)
    return arr


# ---------------------------------------------------------------------------
# Binder générique — pour éviter la répétition
# ---------------------------------------------------------------------------

def _bind_unary(name):
    """Bind une fonction unaire C++ (data,shape,ndim) -> array."""
    if _lib is None:
        return None
    f = getattr(_lib, f"cpp_{name}", None)
    if f is None:
        return None
    f.restype = _c_double_p
    f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                  _c_int64_pp, _c_int_p]

    def wrapper(self_or_x, *args):
        x = args[0] if args else self_or_x
        data, shape, ndim, orig = _arr_info(x)
        out_shape = (ctypes.c_int64 * _MAX_NDIM)()
        out_ndim = ctypes.c_int()
        ptr = f(data, shape, ndim, out_shape, ctypes.byref(out_ndim))
        return _make_result(ptr, out_shape, out_ndim)

    return wrapper


def _bind_unary_f(name, *extra_args_types, default=None):
    """Bind unaire avec paramètre(s) float supplémentaire(s).
    Si default est fourni (liste/tuple), il est utilisé quand l'utilisateur
    ne passe pas explicitement le paramètre correspondant.
    """
    if _lib is None:
        return None
    f = getattr(_lib, f"cpp_{name}", None)
    if f is None:
        return None
    f.restype = _c_double_p
    f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                  _c_int64_pp, _c_int_p] + list(extra_args_types)

    if default is None:
        default = [None] * len(extra_args_types)

    def wrapper(self_or_x, *a):
        x, extra = (a[0], a[1:]) if a else (self_or_x, ())
        # Complète avec les valeurs par défaut si l'utilisateur a omis des args
        while len(extra) < len(extra_args_types):
            extra = extra + (default[len(extra)],)
        # Convertit en ctypes
        extra_ct = []
        for t, v in zip(extra_args_types, extra):
            if v is None:
                raise TypeError(f"{name}: argument requis manquant")
            extra_ct.append(t(v))
        data, shape, ndim, orig = _arr_info(x)
        out_shape = (ctypes.c_int64 * _MAX_NDIM)()
        out_ndim = ctypes.c_int()
        ptr = f(data, shape, ndim, out_shape, ctypes.byref(out_ndim), *extra_ct)
        return _make_result(ptr, out_shape, out_ndim)

    return wrapper


def _bind_unary_i(name):
    """Bind unaire avec paramètre int64."""
    if _lib is None:
        return None
    f = getattr(_lib, f"cpp_{name}", None)
    if f is None:
        return None
    f.restype = _c_double_p
    f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                  _c_int64_pp, _c_int_p, _c_int64]

    def wrapper(self_or_x, *a):
        x, n = (a[0], a[1]) if len(a) >= 2 else (self_or_x, a[0])
        data, shape, ndim, orig = _arr_info(x)
        out_shape = (ctypes.c_int64 * _MAX_NDIM)()
        out_ndim = ctypes.c_int()
        ptr = f(data, shape, ndim, out_shape, ctypes.byref(out_ndim), int(n))
        return _make_result(ptr, out_shape, out_ndim)

    return wrapper


def _bind_binary(name):
    """Bind une fonction binaire C++ (a,b) -> array."""
    if _lib is None:
        return None
    f = getattr(_lib, f"cpp_{name}", None)
    if f is None:
        return None
    f.restype = _c_double_p
    f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                  _c_double_p, _c_int64_p, _c_int,
                  _c_int64_pp, _c_int_p]

    def wrapper(self_or_a, *a):
        a1, a2 = (a[0], a[1]) if len(a) >= 2 else (self_or_a, a[0])
        da, sa, na, _ = _arr_info(a1)
        db, sb, nb, _ = _arr_info(a2)
        out_shape = (ctypes.c_int64 * _MAX_NDIM)()
        out_ndim = ctypes.c_int()
        ptr = f(da, sa, na, db, sb, nb, out_shape, ctypes.byref(out_ndim))
        return _make_result(ptr, out_shape, out_ndim)

    return wrapper


def _bind_scalar(name, extra_types=None):
    """Bind fonction retournant un scalaire double."""
    if _lib is None:
        return None
    f = getattr(_lib, f"cpp_{name}", None)
    if f is None:
        return None
    f.restype = _c_double
    base = [_c_double_p, _c_int64_p, _c_int]
    if extra_types:
        f.argtypes = base + extra_types
    else:
        f.argtypes = base

    def wrapper(self_or_x, *a):
        x, extra = (a[0], a[1:]) if a else (self_or_x, ())
        data, shape, ndim, orig = _arr_info(x)
        if extra:
            return f(data, shape, ndim, *extra)
        return f(data, shape, ndim)

    return wrapper


# ---------------------------------------------------------------------------
# Classe wrapper — expose la même API que l'ancien module pybind11
# ---------------------------------------------------------------------------

class _CppEngine:
    """Wrapper ctypes pour le moteur C++ IA. Même API que le module pybind11."""

    VERSION = "1.0.0"
    BACKEND = "C++"

    # ---- 2. Activations ----
    relu = _bind_unary("relu")
    relu_deriv = _bind_unary("relu_deriv")
    sigmoid = _bind_unary("sigmoid")
    sigmoid_deriv = _bind_unary("sigmoid_deriv")
    tanh = _bind_unary("tanh")
    tanh_deriv = _bind_unary("tanh_deriv")
    leaky_relu = _bind_unary_f("leaky_relu", _c_double, default=[0.01])
    leaky_relu_deriv = _bind_unary_f("leaky_relu_deriv", _c_double)
    softmax = _bind_unary("softmax")
    gelu = _bind_unary("gelu")
    gelu_deriv = _bind_unary("gelu_deriv")

    # ---- 3. Algèbre linéaire ----
    matmul = _bind_binary("matmul")
    add = _bind_binary("add")
    sub = _bind_binary("sub")
    mul = _bind_binary("mul")
    div = _bind_binary("div")
    maximum = _bind_binary("maximum")
    outer = _bind_binary("outer")
    concatenate = _bind_binary("concatenate")

    scale = _bind_unary_f("scale", _c_double)
    add_scalar = _bind_unary_f("add_scalar", _c_double)
    clip = _bind_unary_f("clip", _c_double, _c_double)
    exp = _bind_unary("exp")
    log = _bind_unary("log")
    sqrt = _bind_unary("sqrt")
    pow = _bind_unary_f("pow", _c_double)
    abs = _bind_unary("abs")
    sign = _bind_unary("sign")
    neg = _bind_unary("neg")
    tile = _bind_unary_i("tile")
    repeat = _bind_unary_i("repeat")
    transpose2d = _bind_unary("transpose2d")

    # ---- 4. Convolutions ----
    convolve2d = _bind_binary("convolve2d")

    def convolve2d_backward(self, img, kernel, d_conv):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        di, si, ni, _ = _arr_info(img)
        dk, sk, nk, _ = _arr_info(kernel)
        dd, sd, nd, _ = _arr_info(d_conv)
        f = _lib.cpp_convolve2d_backward
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(di, si, ni, dk, sk, nk, dd, sd, nd, os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def conv1d_forward(self, x, W, b, kernel_size, dilation):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        dx, sx, nx, _ = _arr_info(x)
        dw, sw, nw, _ = _arr_info(W)
        db, sb, nb, _ = _arr_info(b)
        f = _lib.cpp_conv1d_forward
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_int, _c_int,
                      _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(dx, sx, nx, dw, sw, nw, db, sb, nb,
               int(kernel_size), int(dilation), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def conv1d_backward(self, x, W, d_out, kernel_size, dilation):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        dx, sx, nx, _ = _arr_info(x)
        dw, sw, nw, _ = _arr_info(W)
        dd, sd, nd, _ = _arr_info(d_out)
        f = _lib.cpp_conv1d_backward
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_int, _c_int,
                      _c_int64_pp, _c_int_p]
        # 3 output shapes, each up to 8 dims
        out_shapes = (ctypes.c_int64 * (3 * _MAX_NDIM))()
        out_ndims = (ctypes.c_int * 3)()
        ptr = f(dx, sx, nx, dw, sw, nw, dd, sd, nd,
               int(kernel_size), int(dilation),
               out_shapes, out_ndims)
        # Unpack 3 results
        results = []
        offset = 0
        for i in range(3):
            ndim_i = out_ndims[i]
            base = i * _MAX_NDIM
            dims = tuple(out_shapes[base + d] for d in range(ndim_i))
            n = 1
            for d in dims:
                n *= d
            arr = np.empty(n, dtype=np.float64)
            p = ctypes.cast(ptr, ctypes.c_void_p).value + offset * 8
            ctypes.memmove(arr.ctypes.data, p, n * 8)
            arr = arr.reshape(dims)
            results.append(arr)
            offset += n
        _lib.cpp_free(ptr)
        return tuple(results)

    convolve_nd = _bind_binary("convolve_nd")

    def convolve_nd_backward(self, vol, kernel, d_conv):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        dv, sv, nv, _ = _arr_info(vol)
        dk, sk, nk, _ = _arr_info(kernel)
        dd, sd, nd, _ = _arr_info(d_conv)
        f = _lib.cpp_convolve_nd_backward
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(dv, sv, nv, dk, sk, nk, dd, sd, nd, os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    # ---- 5. Normalisation ----
    layer_norm = _bind_unary_f("layer_norm", _c_double, default=[1e-8])

    # ---- 6. Pertes ----
    def mse_loss(self, pred, target):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        dp, sp, np_, _ = _arr_info(pred)
        dt, st, nt, _ = _arr_info(target)
        f = _lib.cpp_mse_loss
        f.restype = _c_double
        f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int]
        return f(dp, sp, np_, dt, st, nt)

    def mse_loss_grad(self, pred, target):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        dp, sp, np_, _ = _arr_info(pred)
        dt, st, nt, _ = _arr_info(target)
        f = _lib.cpp_mse_loss_grad
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(dp, sp, np_, dt, st, nt, os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def cross_entropy_loss(self, logits, target_idx):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        d, s, n, _ = _arr_info(logits)
        f = _lib.cpp_cross_entropy_loss
        f.restype = _c_double
        f.argtypes = [_c_double_p, _c_int64_p, _c_int, _c_int]
        return f(d, s, n, int(target_idx))

    # ---- 7. Initialisation ----
    def _bind_create(self, c_name):
        if _lib is None:
            return None
        f = getattr(_lib, c_name, None)
        if f is None:
            return None
        f.restype = _c_double_p
        f.argtypes = [_c_int64_p, _c_int, _c_int64_pp, _c_int_p]
        return f

    def zeros(self, shape):
        f = self._bind_create("cpp_zeros")
        if f is None:
            raise RuntimeError("C++ non disponible")
        s = tuple(shape)
        cs = (ctypes.c_int64 * len(s))(*s)
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(cs, len(s), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def ones(self, shape):
        f = self._bind_create("cpp_ones")
        if f is None:
            raise RuntimeError("C++ non disponible")
        s = tuple(shape)
        cs = (ctypes.c_int64 * len(s))(*s)
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(cs, len(s), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def xavier_init(self, shape, seed=42):
        f = _lib.cpp_xavier_init
        if f is None:
            raise RuntimeError("C++ non disponible")
        f.restype = _c_double_p
        f.argtypes = [_c_int64_p, _c_int, _c_uint64, _c_int64_pp, _c_int_p]
        s = tuple(shape)
        cs = (ctypes.c_int64 * len(s))(*s)
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(cs, len(s), int(seed), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def randn(self, shape, seed=42):
        f = _lib.cpp_randn
        if f is None:
            raise RuntimeError("C++ non disponible")
        f.restype = _c_double_p
        f.argtypes = [_c_int64_p, _c_int, _c_uint64, _c_int64_pp, _c_int_p]
        s = tuple(shape)
        cs = (ctypes.c_int64 * len(s))(*s)
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(cs, len(s), int(seed), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def uniform(self, shape, lo, hi, seed=42):
        f = _lib.cpp_uniform
        if f is None:
            raise RuntimeError("C++ non disponible")
        f.restype = _c_double_p
        f.argtypes = [_c_int64_p, _c_int, _c_double, _c_double, _c_uint64,
                      _c_int64_pp, _c_int_p]
        s = tuple(shape)
        cs = (ctypes.c_int64 * len(s))(*s)
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(cs, len(s), float(lo), float(hi), int(seed), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def permutation(self, n, seed=42):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        f = _lib.cpp_permutation
        f.restype = _c_int64_p
        f.argtypes = [_c_int64, _c_uint64, _c_int_p]
        out_len = ctypes.c_int()
        ptr = f(int(n), int(seed), ctypes.byref(out_len))
        arr = np.empty(out_len.value, dtype=np.int64)
        ctypes.memmove(arr.ctypes.data, ptr, out_len.value * 8)
        _lib.cpp_free(ptr)
        return arr.tolist()

    # ---- 8. Réductions scalaires ----
    sum = _bind_scalar("sum")
    mean = _bind_scalar("mean")
    max_val = _bind_scalar("max_val")

    def argmax(self, x):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        d, s, n, _ = _arr_info(x)
        f = _lib.cpp_argmax
        f.restype = _c_int64
        f.argtypes = [_c_double_p, _c_int64_p, _c_int]
        return int(f(d, s, n))

    def dot1d(self, a, b):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        da, sa, na, _ = _arr_info(a)
        db, sb, nb, _ = _arr_info(b)
        f = _lib.cpp_dot1d
        f.restype = _c_double
        f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int]
        return f(da, sa, na, db, sb, nb)

    # ---- 8b. Réductions sur axe ----
    def sum_axis(self, x, axis):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        d, s, n, _ = _arr_info(x)
        f = _lib.cpp_sum_axis
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int, _c_int,
                      _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(d, s, n, int(axis), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def mean_axis(self, x, axis):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        d, s, n, _ = _arr_info(x)
        f = _lib.cpp_mean_axis
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int, _c_int,
                      _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(d, s, n, int(axis), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def var_axis(self, x, axis):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        d, s, n, _ = _arr_info(x)
        f = _lib.cpp_var_axis
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int, _c_int,
                      _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(d, s, n, int(axis), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def max_axis(self, x):
        return _bind_unary("max_axis")(x) if _lib else None

    # ---- 9. FFT ----
    def fft_rfft(self, x, n_fft):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        d, s, n, _ = _arr_info(x)
        f = _lib.cpp_fft_rfft
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int, _c_int,
                      _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(d, s, n, int(n_fft), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    # ---- 10. Opérations spécifiques ----
    def add_noise(self, x0, sqrt_alpha, sqrt_one_minus):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        d0, s0, n0, _ = _arr_info(x0)
        d1, s1, n1, _ = _arr_info(sqrt_alpha)
        d2, s2, n2, _ = _arr_info(sqrt_one_minus)
        f = _lib.cpp_add_noise
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(d0, s0, n0, d1, s1, n1, d2, s2, n2, os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def linear(self, x, W, b):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        dx, sx, nx, _ = _arr_info(x)
        dw, sw, nw, _ = _arr_info(W)
        db, sb, nb, _ = _arr_info(b)
        f = _lib.cpp_linear
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(dx, sx, nx, dw, sw, nw, db, sb, nb, os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def ldm_predict_noise(self, x_noisy, class_embedding, class_id,
                           W1, b1, W2, b2):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        dx, sx, nx, _ = _arr_info(x_noisy)
        dc, sc, nc, _ = _arr_info(class_embedding)
        dw1, sw1, nw1, _ = _arr_info(W1)
        db1_, sb1, nb1, _ = _arr_info(b1)
        dw2, sw2, nw2, _ = _arr_info(W2)
        db2_, sb2, nb2, _ = _arr_info(b2)
        f = _lib.cpp_ldm_predict_noise
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_double_p, _c_int64_p, _c_int,
                      _c_int64_pp, _c_int_p, _c_int64_p]
        out_shapes = (ctypes.c_int64 * (4 * _MAX_NDIM))()
        out_ndims = (ctypes.c_int * 4)()
        out_sizes = (ctypes.c_int64 * 4)()
        ptr = f(dx, sx, nx, dc, sc, nc, int(class_id),
               dw1, sw1, nw1, db1_, sb1, nb1,
               dw2, sw2, nw2, db2_, sb2, nb2,
               out_shapes, out_ndims, out_sizes)
        results = []
        offset = 0
        for i in range(4):
            ndim_i = out_ndims[i]
            base = i * _MAX_NDIM
            dims = tuple(out_shapes[base + d] for d in range(ndim_i))
            n = out_sizes[i]
            arr = np.empty(n, dtype=np.float64)
            p = ctypes.cast(ptr, ctypes.c_void_p).value + offset * 8
            ctypes.memmove(arr.ctypes.data, p, n * 8)
            arr = arr.reshape(dims)
            results.append(arr)
            offset += n
        _lib.cpp_free(ptr)
        return tuple(results)

    def pad1d(self, x, target_len):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        d, s, n, _ = _arr_info(x)
        f = _lib.cpp_pad1d
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int, _c_int64,
                      _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(d, s, n, int(target_len), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def diff_axis(self, x, axis):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        d, s, n, _ = _arr_info(x)
        f = _lib.cpp_diff_axis
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int, _c_int,
                      _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(d, s, n, int(axis), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def linspace(self, start, end, num):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        f = _lib.cpp_linspace
        f.restype = _c_double_p
        f.argtypes = [_c_double, _c_double, _c_int64, _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(float(start), float(end), int(num), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def arange(self, start, end, step):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        f = _lib.cpp_arange
        f.restype = _c_double_p
        f.argtypes = [_c_double, _c_double, _c_double, _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(float(start), float(end), float(step), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def mgrid3d(self, size):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        f = _lib.cpp_mgrid3d
        f.restype = _c_double_p
        f.argtypes = [_c_int, _c_int64_pp, _c_int_p, _c_int64_p]
        out_shapes = (ctypes.c_int64 * (3 * _MAX_NDIM))()
        out_ndims = (ctypes.c_int * 3)()
        out_sizes = (ctypes.c_int64 * 3)()
        ptr = f(int(size), out_shapes, out_ndims, out_sizes)
        results = []
        offset = 0
        for i in range(3):
            ndim_i = out_ndims[i]
            base = i * _MAX_NDIM
            dims = tuple(out_shapes[base + d] for d in range(ndim_i))
            n = out_sizes[i]
            arr = np.empty(n, dtype=np.float64)
            p = ctypes.cast(ptr, ctypes.c_void_p).value + offset * 8
            ctypes.memmove(arr.ctypes.data, p, n * 8)
            arr = arr.reshape(dims)
            results.append(arr)
            offset += n
        _lib.cpp_free(ptr)
        return tuple(results)

    def gather(self, x, indices):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        d, s, n, _ = _arr_info(x)
        f = _lib.cpp_gather
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                      _c_int64_p, _c_int,
                      _c_int64_pp, _c_int_p]
        idx = (ctypes.c_int64 * len(indices))(*[int(i) for i in indices])
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(d, s, n, idx, len(indices), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def dct2(self, x, num_coeffs):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        d, s, n, _ = _arr_info(x)
        f = _lib.cpp_dct2
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int, _c_int,
                      _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(d, s, n, int(num_coeffs), os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)

    def histogram(self, x, n_bins, range_lo, range_hi):
        if _lib is None:
            raise RuntimeError("C++ non disponible")
        d, s, n, _ = _arr_info(x)
        f = _lib.cpp_histogram
        f.restype = _c_double_p
        f.argtypes = [_c_double_p, _c_int64_p, _c_int,
                      _c_int, _c_double, _c_double,
                      _c_int64_pp, _c_int_p]
        os = (ctypes.c_int64 * _MAX_NDIM)()
        ond = ctypes.c_int()
        ptr = f(d, s, n, int(n_bins), float(range_lo), float(range_hi),
               os, ctypes.byref(ond))
        return _make_result(ptr, os, ond)


# ---------------------------------------------------------------------------
# Instance singleton
# ---------------------------------------------------------------------------
_engine = _CppEngine()

# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def get_backend():
    return _BACKEND


def is_cpp_available():
    return _lib is not None


def get_core():
    if _lib is None:
        raise RuntimeError("Le moteur C++ n'est pas compile. Lancez: cd IA && make")
    return _engine


# ============================================================================
# Phase 1 : API zero-copy (in-place / into-buffer)
# ============================================================================

def relu_inplace(x):
    """relu modifie x en place. Retourne x (pour chainage). Aucune copie."""
    a = np.ascontiguousarray(x, dtype=np.float64)
    n = a.size
    f = _lib.cpp_relu_inplace
    f.restype = None
    f.argtypes = [_c_double_p, _c_int64]
    f(a.ctypes.data_as(_c_double_p), _c_int64(n))
    # Si x est deja un ndarray float64 contiguous, on a modifie son buffer directement.
    if isinstance(x, np.ndarray) and x.dtype == np.float64 and x.flags['C_CONTIGUOUS']:
        return x
    # Sinon on a travaille sur une copie 'a' : on ecrit le resultat dans x si possible
    if isinstance(x, np.ndarray):
        x[...] = a
    return x


def sigmoid_inplace(x):
    a = np.ascontiguousarray(x, dtype=np.float64)
    n = a.size
    f = _lib.cpp_sigmoid_inplace
    f.restype = None
    f.argtypes = [_c_double_p, _c_int64]
    f(a.ctypes.data_as(_c_double_p), _c_int64(n))
    if isinstance(x, np.ndarray) and x.dtype == np.float64 and x.flags['C_CONTIGUOUS']:
        return x
    if isinstance(x, np.ndarray):
        x[...] = a
    return x


def tanh_inplace(x):
    a = np.ascontiguousarray(x, dtype=np.float64)
    n = a.size
    f = _lib.cpp_tanh_inplace
    f.restype = None
    f.argtypes = [_c_double_p, _c_int64]
    f(a.ctypes.data_as(_c_double_p), _c_int64(n))
    if isinstance(x, np.ndarray) and x.dtype == np.float64 and x.flags['C_CONTIGUOUS']:
        return x
    if isinstance(x, np.ndarray):
        x[...] = a
    return x


def add_inplace(a, b):
    """a += b, en place dans a. Aucune copie si a est float64 contiguous."""
    a_arr = np.ascontiguousarray(a, dtype=np.float64)
    b_arr = np.ascontiguousarray(b, dtype=np.float64)
    if a_arr.size != b_arr.size:
        raise ValueError(f"add_inplace: size mismatch {a_arr.size} vs {b_arr.size}")
    f = _lib.cpp_add_inplace
    f.restype = None
    f.argtypes = [_c_double_p, _c_double_p, _c_int64]
    f(a_arr.ctypes.data_as(_c_double_p),
      b_arr.ctypes.data_as(_c_double_p),
      _c_int64(a_arr.size))
    if isinstance(a, np.ndarray) and a.dtype == np.float64 and a.flags['C_CONTIGUOUS']:
        return a
    if isinstance(a, np.ndarray):
        a[...] = a_arr
    return a


def scale_inplace(a, s):
    """a *= s, en place dans a."""
    a_arr = np.ascontiguousarray(a, dtype=np.float64)
    f = _lib.cpp_scale_inplace
    f.restype = None
    f.argtypes = [_c_double_p, _c_int64, ctypes.c_double]
    f(a_arr.ctypes.data_as(_c_double_p), _c_int64(a_arr.size), ctypes.c_double(float(s)))
    if isinstance(a, np.ndarray) and a.dtype == np.float64 and a.flags['C_CONTIGUOUS']:
        return a
    if isinstance(a, np.ndarray):
        a[...] = a_arr
    return a


def matmul_zero_copy(a, b, out=None):
    """matmul 2D sans aucune copie buffer.
    A[M,K] @ B[K,N] -> C[M,N]. Si out est None, on l'alloue.
    Si out est fourni (float64 contiguous shape (M,N)), on ecrit dedans.
    """
    a_arr = np.ascontiguousarray(a, dtype=np.float64)
    b_arr = np.ascontiguousarray(b, dtype=np.float64)
    if a_arr.ndim != 2 or b_arr.ndim != 2:
        raise ValueError("matmul_zero_copy: requires 2D inputs")
    M, K = a_arr.shape
    K2, N = b_arr.shape
    if K != K2:
        raise ValueError(f"matmul_zero_copy: K mismatch {K} vs {K2}")
    if out is None:
        out = np.empty((M, N), dtype=np.float64)
    elif not (isinstance(out, np.ndarray) and out.dtype == np.float64
              and out.flags['C_CONTIGUOUS'] and out.shape == (M, N)):
        out = np.ascontiguousarray(out, dtype=np.float64).reshape(M, N)
    f = _lib.cpp_matmul_2d_into
    f.restype = None
    f.argtypes = [_c_double_p, _c_double_p, _c_double_p, _c_int64, _c_int64, _c_int64]
    f(a_arr.ctypes.data_as(_c_double_p),
      b_arr.ctypes.data_as(_c_double_p),
      out.ctypes.data_as(_c_double_p),
      _c_int64(M), _c_int64(K), _c_int64(N))
    return out


def convolve2d_zero_copy(img, kernel, out=None):
    """convolve2d sans aucune copie buffer.
    img[ih,iw] * kernel[kh,kw] -> out[ih-kh+1, iw-kw+1].
    """
    img_arr = np.ascontiguousarray(img, dtype=np.float64)
    k_arr = np.ascontiguousarray(kernel, dtype=np.float64)
    if img_arr.ndim != 2 or k_arr.ndim != 2:
        raise ValueError("convolve2d_zero_copy: requires 2D inputs")
    ih, iw = img_arr.shape
    kh, kw = k_arr.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    if out is None:
        out = np.empty((oh, ow), dtype=np.float64)
    elif not (isinstance(out, np.ndarray) and out.dtype == np.float64
              and out.flags['C_CONTIGUOUS'] and out.shape == (oh, ow)):
        out = np.ascontiguousarray(out, dtype=np.float64).reshape(oh, ow)
    f = _lib.cpp_convolve2d_into
    f.restype = None
    f.argtypes = [_c_double_p, _c_int64, _c_int64,
                  _c_double_p, _c_int64, _c_int64, _c_double_p]
    f(img_arr.ctypes.data_as(_c_double_p), _c_int64(ih), _c_int64(iw),
      k_arr.ctypes.data_as(_c_double_p), _c_int64(kh), _c_int64(kw),
      out.ctypes.data_as(_c_double_p))
    return out


# ============================================================================
# Phase 4 : Autograd tape-based (API Python)
# ============================================================================

class AutogradVar:
    """Wrapper Python sur un Var du tape C++. L'id est gere cote C++."""
    __slots__ = ('id',)

    def __init__(self, id_):
        self.id = id_

    @classmethod
    def from_array(cls, arr):
        """Cree une feuille a partir d'un numpy array."""
        a = np.ascontiguousarray(arr, dtype=np.float64)
        shape = (ctypes.c_int64 * a.ndim)(*a.shape)
        f = _lib.ag_make_var
        f.restype = ctypes.c_int64
        f.argtypes = [_c_double_p, _c_int64_p, ctypes.c_int]
        vid = f(a.ctypes.data_as(_c_double_p), shape, ctypes.c_int(a.ndim))
        return cls(vid)

    def _data_ptr(self):
        """Retourne (numpy_array_vue, ptr, shape, ndim).
        La vue reste valide tant que le tape C++ n'est pas reset.
        """
        out_shape = (ctypes.c_int64 * _MAX_NDIM)()
        out_ndim = ctypes.c_int()
        f = _lib.ag_get_data_ptr
        f.restype = _c_double_p
        f.argtypes = [ctypes.c_int64, _c_int64_p, _c_int_p]
        ptr = f(ctypes.c_int64(self.id), out_shape, ctypes.byref(out_ndim))
        if not ptr:
            raise RuntimeError("Var data inaccessible")
        ndim = out_ndim.value
        dims = tuple(out_shape[i] for i in range(ndim))
        n = 1
        for d in dims: n *= d
        # On construit un numpy array pointant directement vers le buffer C++.
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, _c_double_p), shape=(n,))
        return arr.reshape(dims), ptr, dims, ndim

    def data(self):
        """Retourne une copie numpy de la valeur."""
        arr, _, _, _ = self._data_ptr()
        return arr.copy()

    def grad(self):
        """Retourne une copie numpy du gradient (None si pas encore backward)."""
        out_shape = (ctypes.c_int64 * _MAX_NDIM)()
        out_ndim = ctypes.c_int()
        f = _lib.ag_get_grad_ptr
        f.restype = _c_double_p
        f.argtypes = [ctypes.c_int64, _c_int64_p, _c_int_p]
        ptr = f(ctypes.c_int64(self.id), out_shape, ctypes.byref(out_ndim))
        if not ptr:
            return None
        ndim = out_ndim.value
        dims = tuple(out_shape[i] for i in range(ndim))
        n = 1
        for d in dims: n *= d
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, _c_double_p), shape=(n,))
        return arr.reshape(dims).copy()

    def backward(self):
        f = _lib.ag_backward
        f.restype = None
        f.argtypes = [ctypes.c_int64]
        f(ctypes.c_int64(self.id))


def ag_reset():
    f = _lib.ag_reset
    f.restype = None
    f.argtypes = []
    f()


def ag_tape_size():
    f = _lib.ag_tape_size
    f.restype = ctypes.c_int64
    f.argtypes = []
    return f()


def _ag_unary_op(name):
    f = getattr(_lib, f"ag_{name}", None)
    if f is None:
        return None
    f.restype = ctypes.c_int64
    f.argtypes = [ctypes.c_int64]
    def wrapper(x):
        if not isinstance(x, AutogradVar):
            raise TypeError(f"ag_{name}: expected AutogradVar, got {type(x)}")
        return AutogradVar(f(ctypes.c_int64(x.id)))
    return wrapper


def _ag_binary_op(name):
    f = getattr(_lib, f"ag_{name}", None)
    if f is None:
        return None
    f.restype = ctypes.c_int64
    f.argtypes = [ctypes.c_int64, ctypes.c_int64]
    def wrapper(a, b):
        if not isinstance(a, AutogradVar) or not isinstance(b, AutogradVar):
            raise TypeError(f"ag_{name}: expected AutogradVars")
        return AutogradVar(f(ctypes.c_int64(a.id), ctypes.c_int64(b.id)))
    return wrapper


def _ag_scale():
    if _lib is None:
        return None
    f = getattr(_lib, "ag_scale", None)
    if f is None:
        return None
    f.restype = ctypes.c_int64
    f.argtypes = [ctypes.c_int64, ctypes.c_double]
    def wrapper(x, s):
        if not isinstance(x, AutogradVar):
            raise TypeError("ag_scale: expected AutogradVar")
        return AutogradVar(f(ctypes.c_int64(x.id), ctypes.c_double(float(s))))
    return wrapper


# Constructeurs d'ops autograd exposes au module
ag_relu = _ag_unary_op("relu")
ag_sigmoid = _ag_unary_op("sigmoid")
ag_tanh = _ag_unary_op("tanh")
ag_add = _ag_binary_op("add")
ag_sub = _ag_binary_op("sub")
ag_mul = _ag_binary_op("mul")
ag_matmul = _ag_binary_op("matmul")
ag_mse = _ag_binary_op("mse")
ag_dot = _ag_binary_op("dot")
ag_scale = _ag_scale()


__all__ = [
    "get_backend", "is_cpp_available", "get_core", "_BACKEND",
    # Phase 1
    "relu_inplace", "sigmoid_inplace", "tanh_inplace",
    "add_inplace", "scale_inplace",
    "matmul_zero_copy", "convolve2d_zero_copy",
    # Phase 4
    "AutogradVar", "ag_reset", "ag_tape_size",
    "ag_relu", "ag_sigmoid", "ag_tanh",
    "ag_add", "ag_sub", "ag_mul", "ag_matmul", "ag_mse", "ag_dot", "ag_scale",
]