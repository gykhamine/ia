/**
 * IA/cpp/c_api.cpp — API C pure (extern "C") pour le moteur IA.
 *
 * AUCUNE dépendance externe (pas pybind11, pas Python.h).
 * Compile en .so pur C++, chargé via ctypes en Python.
 *
 * Convention:
 *   - Fonctions retournant un tableau: allouent via malloc, caller doit cpp_free()
 *   - out_shape / out_ndim: remplis par la fonction pour les sorties tableau
 *   - Fonctions retournant un scalaire: retour direct double / int64_t
 */

#include "engine.h"
#include "autograd.h"
#include <cstdlib>
#include <cstring>

using namespace ia_core;

// ============================================================================
// Helpers internes
// ============================================================================

static Tensor make_tensor(const double* data, const int64_t* shape, int ndim) {
    std::vector<int64_t> dims(shape, shape + ndim);
    int64_t total = 1;
    for (auto d : dims) total *= d;
    std::vector<double> vec(data, data + total);
    return Tensor(std::move(vec), Shape(dims));
}

/** Copie les données d'un Tensor dans un buffer malloc, remplit shape/ndim. */
static double* return_tensor(const Tensor& t, int64_t* out_shape, int* out_ndim) {
    *out_ndim = static_cast<int>(t.ndim());
    for (int i = 0; i < *out_ndim; ++i) out_shape[i] = t.shape[i];
    int64_t n = t.size();
    double* buf = static_cast<double*>(std::malloc(n * sizeof(double)));
    std::memcpy(buf, t.data.data(), n * sizeof(double));
    return buf;
}

// ============================================================================
// Memory
// ============================================================================

extern "C" void cpp_free(void* ptr) { std::free(ptr); }

// ============================================================================
// Macros pour ops unaires simples (même shape en sortie)
// ============================================================================

#define UNARY_OP(name) \
    extern "C" double* cpp_##name(const double* data, const int64_t* shape, int ndim, \
                                   int64_t* out_shape, int* out_ndim) { \
        auto in = make_tensor(data, shape, ndim); \
        auto out = ia_core::name(in); \
        return return_tensor(out, out_shape, out_ndim); \
    }

#define UNARY_OP_SCALAR1(name, param_type, param_name) \
    extern "C" double* cpp_##name(const double* data, const int64_t* shape, int ndim, \
                                   int64_t* out_shape, int* out_ndim, param_type param_name) { \
        auto in = make_tensor(data, shape, ndim); \
        auto out = ia_core::name(in, param_name); \
        return return_tensor(out, out_shape, out_ndim); \
    }

#define UNARY_OP_SCALAR2(name, p1type, p1name, p2type, p2name) \
    extern "C" double* cpp_##name(const double* data, const int64_t* shape, int ndim, \
                                   int64_t* out_shape, int* out_ndim, \
                                   p1type p1name, p2type p2name) { \
        auto in = make_tensor(data, shape, ndim); \
        auto out = ia_core::name(in, p1name, p2name); \
        return return_tensor(out, out_shape, out_ndim); \
    }

// ============================================================================
// Macros pour ops binaires élémentaires (même shape)
// ============================================================================

#define BINARY_OP(name) \
    extern "C" double* cpp_##name( \
        const double* a, const int64_t* a_shape, int a_ndim, \
        const double* b, const int64_t* b_shape, int b_ndim, \
        int64_t* out_shape, int* out_ndim) { \
        auto ta = make_tensor(a, a_shape, a_ndim); \
        auto tb = make_tensor(b, b_shape, b_ndim); \
        auto out = ia_core::name(ta, tb); \
        return return_tensor(out, out_shape, out_ndim); \
    }

// ============================================================================
// 2. Activations
// ============================================================================

UNARY_OP(relu)
UNARY_OP(relu_deriv)
UNARY_OP(sigmoid)
UNARY_OP(sigmoid_deriv)
UNARY_OP(tanh_act)
UNARY_OP(tanh_deriv)
UNARY_OP_SCALAR1(leaky_relu, double, alpha)
UNARY_OP_SCALAR1(leaky_relu_deriv, double, alpha)
UNARY_OP(softmax)
UNARY_OP(gelu)
UNARY_OP(gelu_deriv)

// tanh alias
extern "C" double* cpp_tanh(const double* data, const int64_t* shape, int ndim,
                             int64_t* out_shape, int* out_ndim) {
    auto in = make_tensor(data, shape, ndim);
    auto out = ia_core::tanh_act(in);
    return return_tensor(out, out_shape, out_ndim);
}

// ============================================================================
// 3. Algèbre linéaire — élémentaire
// ============================================================================

BINARY_OP(add)
BINARY_OP(sub)
BINARY_OP(mul)
BINARY_OP(div)
BINARY_OP(maximum)
BINARY_OP(matmul)
BINARY_OP(outer)
BINARY_OP(concatenate)

extern "C" double* cpp_scale(const double* data, const int64_t* shape, int ndim,
                              int64_t* out_shape, int* out_ndim, double s) {
    auto in = make_tensor(data, shape, ndim);
    auto out = ia_core::scale(in, s);
    return return_tensor(out, out_shape, out_ndim);
}

extern "C" double* cpp_add_scalar(const double* data, const int64_t* shape, int ndim,
                                   int64_t* out_shape, int* out_ndim, double s) {
    auto in = make_tensor(data, shape, ndim);
    auto out = ia_core::add_scalar(in, s);
    return return_tensor(out, out_shape, out_ndim);
}

UNARY_OP_SCALAR2(clip, double, lo, double, hi)
UNARY_OP(exp)
UNARY_OP(log)
UNARY_OP(sqrt)

extern "C" double* cpp_pow(const double* data, const int64_t* shape, int ndim,
                            int64_t* out_shape, int* out_ndim, double n) {
    auto in = make_tensor(data, shape, ndim);
    auto out = ia_core::pow(in, n);
    return return_tensor(out, out_shape, out_ndim);
}

UNARY_OP(abs)
UNARY_OP(sign)
UNARY_OP(neg)

extern "C" double* cpp_tile(const double* data, const int64_t* shape, int ndim,
                             int64_t* out_shape, int* out_ndim, int64_t n) {
    auto in = make_tensor(data, shape, ndim);
    auto out = ia_core::tile(in, n);
    return return_tensor(out, out_shape, out_ndim);
}

extern "C" double* cpp_repeat(const double* data, const int64_t* shape, int ndim,
                               int64_t* out_shape, int* out_ndim, int64_t n) {
    auto in = make_tensor(data, shape, ndim);
    auto out = ia_core::repeat(in, n);
    return return_tensor(out, out_shape, out_ndim);
}

// ============================================================================
// 4. Convolutions
// ============================================================================

BINARY_OP(convolve2d)

extern "C" double* cpp_convolve2d_backward(
    const double* img, const int64_t* img_shape, int img_ndim,
    const double* kernel, const int64_t* k_shape, int k_ndim,
    const double* d_conv, const int64_t* dc_shape, int dc_ndim,
    int64_t* out_shape, int* out_ndim) {
    auto ti = make_tensor(img, img_shape, img_ndim);
    auto tk = make_tensor(kernel, k_shape, k_ndim);
    auto td = make_tensor(d_conv, dc_shape, dc_ndim);
    auto out = ia_core::convolve2d_backward(ti, tk, td);
    return return_tensor(out, out_shape, out_ndim);
}

extern "C" double* cpp_conv1d_forward(
    const double* x, const int64_t* x_shape, int x_ndim,
    const double* W, const int64_t* W_shape, int W_ndim,
    const double* b, const int64_t* b_shape, int b_ndim,
    int kernel_size, int dilation,
    int64_t* out_shape, int* out_ndim) {
    auto tx = make_tensor(x, x_shape, x_ndim);
    auto tW = make_tensor(W, W_shape, W_ndim);
    auto tb = make_tensor(b, b_shape, b_ndim);
    auto out = ia_core::conv1d_forward(tx, tW, tb, kernel_size, dilation);
    return return_tensor(out, out_shape, out_ndim);
}

// conv1d_backward returns 3 tensors — we pack them sequentially:
// [d_x data | d_W data | d_b data] with shapes written to out_shapes (3x max_ndim)
extern "C" double* cpp_conv1d_backward(
    const double* x, const int64_t* x_shape, int x_ndim,
    const double* W, const int64_t* W_shape, int W_ndim,
    const double* d_out, const int64_t* do_shape, int do_ndim,
    int kernel_size, int dilation,
    int64_t* out_shapes, int* out_ndims) {
    auto tx = make_tensor(x, x_shape, x_ndim);
    auto tW = make_tensor(W, W_shape, W_ndim);
    auto td = make_tensor(d_out, do_shape, do_ndim);
    auto results = ia_core::conv1d_backward(tx, tW, td, kernel_size, dilation);
    // Pack 3 results into one buffer
    int64_t total = results[0].size() + results[1].size() + results[2].size();
    double* buf = static_cast<double*>(std::malloc(total * sizeof(double)));
    int64_t offset = 0;
    for (int i = 0; i < 3; ++i) {
        std::memcpy(buf + offset, results[i].data.data(), results[i].size() * sizeof(double));
        out_ndims[i] = static_cast<int>(results[i].ndim());
        int base = i * 8;
        for (int d = 0; d < results[i].ndim(); ++d) out_shapes[base + d] = results[i].shape[d];
        offset += results[i].size();
    }
    return buf;
}

BINARY_OP(convolve_nd)

extern "C" double* cpp_convolve_nd_backward(
    const double* vol, const int64_t* vol_shape, int vol_ndim,
    const double* kernel, const int64_t* k_shape, int k_ndim,
    const double* d_conv, const int64_t* dc_shape, int dc_ndim,
    int64_t* out_shape, int* out_ndim) {
    auto tv = make_tensor(vol, vol_shape, vol_ndim);
    auto tk = make_tensor(kernel, k_shape, k_ndim);
    auto td = make_tensor(d_conv, dc_shape, dc_ndim);
    auto out = ia_core::convolve_nd_backward(tv, tk, td);
    return return_tensor(out, out_shape, out_ndim);
}

// ============================================================================
// 5. Normalisation
// ============================================================================

UNARY_OP_SCALAR1(layer_norm, double, eps)

// ============================================================================
// 6. Pertes
// ============================================================================

extern "C" double cpp_mse_loss(
    const double* pred, const int64_t* pred_shape, int pred_ndim,
    const double* target, const int64_t* target_shape, int target_ndim) {
    auto tp = make_tensor(pred, pred_shape, pred_ndim);
    auto tt = make_tensor(target, target_shape, target_ndim);
    return ia_core::mse_loss(tp, tt);
}

extern "C" double* cpp_mse_loss_grad(
    const double* pred, const int64_t* pred_shape, int pred_ndim,
    const double* target, const int64_t* target_shape, int target_ndim,
    int64_t* out_shape, int* out_ndim) {
    auto tp = make_tensor(pred, pred_shape, pred_ndim);
    auto tt = make_tensor(target, target_shape, target_ndim);
    auto out = ia_core::mse_loss_grad(tp, tt);
    return return_tensor(out, out_shape, out_ndim);
}

extern "C" double cpp_cross_entropy_loss(
    const double* logits, const int64_t* logits_shape, int logits_ndim,
    int target_idx) {
    auto t = make_tensor(logits, logits_shape, logits_ndim);
    return ia_core::cross_entropy_loss(t, target_idx);
}

// ============================================================================
// 7. Initialisation
// ============================================================================

extern "C" double* cpp_zeros(const int64_t* shape, int ndim,
                              int64_t* out_shape, int* out_ndim) {
    std::vector<int64_t> dims(shape, shape + ndim);
    auto t = ia_core::zeros(Shape(dims));
    return return_tensor(t, out_shape, out_ndim);
}

extern "C" double* cpp_ones(const int64_t* shape, int ndim,
                             int64_t* out_shape, int* out_ndim) {
    std::vector<int64_t> dims(shape, shape + ndim);
    auto t = ia_core::ones(Shape(dims));
    return return_tensor(t, out_shape, out_ndim);
}

extern "C" double* cpp_xavier_init(const int64_t* shape, int ndim, uint64_t seed,
                                    int64_t* out_shape, int* out_ndim) {
    std::vector<int64_t> dims(shape, shape + ndim);
    auto t = ia_core::xavier_init(Shape(dims), seed);
    return return_tensor(t, out_shape, out_ndim);
}

extern "C" double* cpp_randn(const int64_t* shape, int ndim, uint64_t seed,
                              int64_t* out_shape, int* out_ndim) {
    std::vector<int64_t> dims(shape, shape + ndim);
    auto t = ia_core::randn(Shape(dims), seed);
    return return_tensor(t, out_shape, out_ndim);
}

extern "C" double* cpp_uniform(const int64_t* shape, int ndim,
                                double lo, double hi, uint64_t seed,
                                int64_t* out_shape, int* out_ndim) {
    std::vector<int64_t> dims(shape, shape + ndim);
    auto t = ia_core::uniform(Shape(dims), lo, hi, seed);
    return return_tensor(t, out_shape, out_ndim);
}

extern "C" int64_t* cpp_permutation(int64_t n, uint64_t seed, int* out_len) {
    auto idx = ia_core::permutation(n, seed);
    *out_len = static_cast<int>(idx.size());
    int64_t* buf = static_cast<int64_t*>(std::malloc(idx.size() * sizeof(int64_t)));
    std::memcpy(buf, idx.data(), idx.size() * sizeof(int64_t));
    return buf;
}

// ============================================================================
// 8. Réductions scalaires
// ============================================================================

extern "C" double cpp_sum(const double* data, const int64_t* shape, int ndim) {
    return ia_core::sum(make_tensor(data, shape, ndim));
}

extern "C" double cpp_mean(const double* data, const int64_t* shape, int ndim) {
    return ia_core::mean(make_tensor(data, shape, ndim));
}

extern "C" double cpp_max_val(const double* data, const int64_t* shape, int ndim) {
    return ia_core::max_val(make_tensor(data, shape, ndim));
}

extern "C" int64_t cpp_argmax(const double* data, const int64_t* shape, int ndim) {
    return ia_core::argmax(make_tensor(data, shape, ndim));
}

extern "C" double cpp_dot1d(
    const double* a, const int64_t* a_shape, int a_ndim,
    const double* b, const int64_t* b_shape, int b_ndim) {
    return ia_core::dot1d(make_tensor(a, a_shape, a_ndim),
                          make_tensor(b, b_shape, b_ndim));
}

// ============================================================================
// 8b. Réductions sur axe (retournent un tableau)
// ============================================================================

extern "C" double* cpp_sum_axis(const double* data, const int64_t* shape, int ndim,
                                 int axis, int64_t* out_shape, int* out_ndim) {
    auto t = make_tensor(data, shape, ndim);
    auto out = ia_core::sum_axis(t, axis);
    return return_tensor(out, out_shape, out_ndim);
}

extern "C" double* cpp_mean_axis(const double* data, const int64_t* shape, int ndim,
                                  int axis, int64_t* out_shape, int* out_ndim) {
    auto t = make_tensor(data, shape, ndim);
    auto out = ia_core::mean_axis(t, axis);
    return return_tensor(out, out_shape, out_ndim);
}

extern "C" double* cpp_var_axis(const double* data, const int64_t* shape, int ndim,
                                 int axis, int64_t* out_shape, int* out_ndim) {
    auto t = make_tensor(data, shape, ndim);
    auto out = ia_core::var_axis(t, axis);
    return return_tensor(out, out_shape, out_ndim);
}

extern "C" double* cpp_max_axis(const double* data, const int64_t* shape, int ndim,
                                 int64_t* out_shape, int* out_ndim) {
    auto t = make_tensor(data, shape, ndim);
    auto out = ia_core::max_axis(t);
    return return_tensor(out, out_shape, out_ndim);
}

// ============================================================================
// 9. FFT
// ============================================================================

extern "C" double* cpp_fft_rfft(const double* data, const int64_t* shape, int ndim,
                                 int n_fft, int64_t* out_shape, int* out_ndim) {
    auto t = make_tensor(data, shape, ndim);
    auto out = ia_core::fft_rfft(t, n_fft);
    return return_tensor(out, out_shape, out_ndim);
}

// ============================================================================
// 10. Opérations spécifiques
// ============================================================================

// add_noise: 3 arrays
extern "C" double* cpp_add_noise(
    const double* x0, const int64_t* x0_shape, int x0_ndim,
    const double* sa, const int64_t* sa_shape, int sa_ndim,
    const double* som, const int64_t* som_shape, int som_ndim,
    int64_t* out_shape, int* out_ndim) {
    auto t1 = make_tensor(x0, x0_shape, x0_ndim);
    auto t2 = make_tensor(sa, sa_shape, sa_ndim);
    auto t3 = make_tensor(som, som_shape, som_ndim);
    auto out = ia_core::add_noise(t1, t2, t3);
    return return_tensor(out, out_shape, out_ndim);
}

// linear: x @ W + b
extern "C" double* cpp_linear(
    const double* x, const int64_t* x_shape, int x_ndim,
    const double* W, const int64_t* W_shape, int W_ndim,
    const double* b, const int64_t* b_shape, int b_ndim,
    int64_t* out_shape, int* out_ndim) {
    auto tx = make_tensor(x, x_shape, x_ndim);
    auto tW = make_tensor(W, W_shape, W_ndim);
    auto tb = make_tensor(b, b_shape, b_ndim);
    auto out = ia_core::linear(tx, tW, tb);
    return return_tensor(out, out_shape, out_ndim);
}

// ldm_predict_noise: complex, returns 4 tensors packed
extern "C" double* cpp_ldm_predict_noise(
    const double* x_noisy, const int64_t* xn_shape, int xn_ndim,
    const double* class_emb, const int64_t* ce_shape, int ce_ndim,
    int class_id,
    const double* W1, const int64_t* W1_shape, int W1_ndim,
    const double* b1, const int64_t* b1_shape, int b1_ndim,
    const double* W2, const int64_t* W2_shape, int W2_ndim,
    const double* b2, const int64_t* b2_shape, int b2_ndim,
    int64_t* out_shapes, int* out_ndims, int64_t* out_sizes) {
    auto txn = make_tensor(x_noisy, xn_shape, xn_ndim);
    auto tce = make_tensor(class_emb, ce_shape, ce_ndim);
    auto tW1 = make_tensor(W1, W1_shape, W1_ndim);
    auto tb1 = make_tensor(b1, b1_shape, b1_ndim);
    auto tW2 = make_tensor(W2, W2_shape, W2_ndim);
    auto tb2 = make_tensor(b2, b2_shape, b2_ndim);
    auto r = ia_core::ldm_predict_noise(txn, tce, class_id, tW1, tb1, tW2, tb2);
    Tensor parts[4] = {r.output, r.x_concat, r.z1, r.h1};
    int64_t total = 0;
    for (int i = 0; i < 4; ++i) {
        out_ndims[i] = static_cast<int>(parts[i].ndim());
        int base = i * 8;
        for (int d = 0; d < parts[i].ndim(); ++d) out_shapes[base + d] = parts[i].shape[d];
        out_sizes[i] = parts[i].size();
        total += parts[i].size();
    }
    double* buf = static_cast<double*>(std::malloc(total * sizeof(double)));
    int64_t off = 0;
    for (int i = 0; i < 4; ++i) {
        std::memcpy(buf + off, parts[i].data.data(), parts[i].size() * sizeof(double));
        off += parts[i].size();
    }
    return buf;
}

// pad1d
extern "C" double* cpp_pad1d(const double* data, const int64_t* shape, int ndim,
                              int64_t target_len,
                              int64_t* out_shape, int* out_ndim) {
    auto t = make_tensor(data, shape, ndim);
    auto out = ia_core::pad1d(t, target_len);
    return return_tensor(out, out_shape, out_ndim);
}

// diff_axis
extern "C" double* cpp_diff_axis(const double* data, const int64_t* shape, int ndim,
                                  int axis, int64_t* out_shape, int* out_ndim) {
    auto t = make_tensor(data, shape, ndim);
    auto out = ia_core::diff_axis(t, axis);
    return return_tensor(out, out_shape, out_ndim);
}

// linspace
extern "C" double* cpp_linspace(double start, double end, int64_t num,
                                 int64_t* out_shape, int* out_ndim) {
    auto t = ia_core::linspace(start, end, num);
    return return_tensor(t, out_shape, out_ndim);
}

// arange
extern "C" double* cpp_arange(double start, double end, double step,
                               int64_t* out_shape, int* out_ndim) {
    auto t = ia_core::arange(start, end, step);
    return return_tensor(t, out_shape, out_ndim);
}

// mgrid3d: returns 3 tensors packed
extern "C" double* cpp_mgrid3d(int size,
                                int64_t* out_shapes, int* out_ndims, int64_t* out_sizes) {
    auto grids = ia_core::mgrid3d(size);
    int64_t total = 0;
    for (int i = 0; i < 3; ++i) {
        out_ndims[i] = static_cast<int>(grids[i].ndim());
        int base = i * 8;
        for (int d = 0; d < grids[i].ndim(); ++d) out_shapes[base + d] = grids[i].shape[d];
        out_sizes[i] = grids[i].size();
        total += grids[i].size();
    }
    double* buf = static_cast<double*>(std::malloc(total * sizeof(double)));
    int64_t off = 0;
    for (int i = 0; i < 3; ++i) {
        std::memcpy(buf + off, grids[i].data.data(), grids[i].size() * sizeof(double));
        off += grids[i].size();
    }
    return buf;
}

// gather
extern "C" double* cpp_gather(const double* data, const int64_t* shape, int ndim,
                               const int64_t* indices, int n_indices,
                               int64_t* out_shape, int* out_ndim) {
    auto t = make_tensor(data, shape, ndim);
    std::vector<int64_t> idx(indices, indices + n_indices);
    auto out = ia_core::gather(t, idx);
    return return_tensor(out, out_shape, out_ndim);
}

// dct2
extern "C" double* cpp_dct2(const double* data, const int64_t* shape, int ndim,
                             int num_coeffs, int64_t* out_shape, int* out_ndim) {
    auto t = make_tensor(data, shape, ndim);
    auto out = ia_core::dct2(t, num_coeffs);
    return return_tensor(out, out_shape, out_ndim);
}

// histogram: returns double array (counts)
extern "C" double* cpp_histogram(const double* data, const int64_t* shape, int ndim,
                                  int n_bins, double range_lo, double range_hi,
                                  int64_t* out_shape, int* out_ndim) {
    auto t = make_tensor(data, shape, ndim);
    auto h = ia_core::histogram(t, n_bins, range_lo, range_hi);
    *out_ndim = 1;
    out_shape[0] = static_cast<int64_t>(h.size());
    double* buf = static_cast<double*>(std::malloc(h.size() * sizeof(double)));
    std::memcpy(buf, h.data(), h.size() * sizeof(double));
    return buf;
}

// transpose2d
extern "C" double* cpp_transpose2d(const double* data, const int64_t* shape, int ndim,
                                    int64_t* out_shape, int* out_ndim) {
    auto t = make_tensor(data, shape, ndim);
    auto out = t.transpose2d();
    return return_tensor(out, out_shape, out_ndim);
}

// ============================================================================
// Phase 1 : API zero-copy (in-place / into-buffer)
//   Pas de malloc, pas de retour via return_tensor. Le buffer est fourni par
//   l'appelant (typiquement: le buffer numpy lui-meme via ctypes).
// ============================================================================

extern "C" void cpp_relu_inplace(double* data, int64_t n) {
    ia_core::relu_inplace(data, n);
}

extern "C" void cpp_sigmoid_inplace(double* data, int64_t n) {
    ia_core::sigmoid_inplace(data, n);
}

extern "C" void cpp_tanh_inplace(double* data, int64_t n) {
    ia_core::tanh_inplace(data, n);
}

extern "C" void cpp_add_inplace(double* a, const double* b, int64_t n) {
    ia_core::add_inplace(a, b, n);
}

extern "C" void cpp_axpy_inplace(double* a, const double* b, int64_t n, double s) {
    ia_core::axpy_inplace(a, b, n, s);
}

extern "C" void cpp_scale_inplace(double* a, int64_t n, double s) {
    ia_core::scale_inplace(a, n, s);
}

/** matmul 2D zero-copy : A[M,K] @ B[K,N] -> C[M,N] (deja alloue par l'appelant). */
extern "C" void cpp_matmul_2d_into(const double* A, const double* B, double* C,
                                   int64_t M, int64_t K, int64_t N) {
    ia_core::matmul_2d_into(A, M, K, B, N, C);
}

/** convolve2d zero-copy : img[ih*iw] * kernel[kh*kw] -> out[(ih-kh+1)*(iw-kw+1)]. */
extern "C" void cpp_convolve2d_into(const double* img, int64_t ih, int64_t iw,
                                    const double* kernel, int64_t kh, int64_t kw,
                                    double* out) {
    ia_core::convolve2d_into(img, ih, iw, kernel, kh, kw, out);
}

// ============================================================================
// Phase 4 : API C pour l'autograd tape-based
//
// Le tape est thread_local cote C++. Les Vars sont identifiees par un int64_t.
// Le Python wrapper garde les id et appelle ces fonctions.
// ============================================================================

/** Cree une feuille (Var sans parent) et retourne son id. */
extern "C" int64_t ag_make_var(const double* data, const int64_t* shape, int ndim) {
    auto t = make_tensor(data, shape, ndim);
    ia_autograd::Var v(std::move(t));
    return v.id;
}

/** Remet le tape a zero. */
extern "C" void ag_reset() {
    ia_autograd::reset_tape();
}

/** Retourne le nombre de noeuds sur le tape (debug). */
extern "C" int64_t ag_tape_size() {
    return ia_autograd::tape_size();
}

/** Pointeur direct vers le buffer data d'un Var (NE PAS LIBERER).
 *  Le pointeur reste valide tant que le tape n'est pas reset. */
extern "C" const double* ag_get_data_ptr(int64_t id, int64_t* out_shape, int* out_ndim) {
    return ag_get_var_data_ptr(id, out_shape, out_ndim);
}

/** Pointeur direct vers le buffer grad d'un Var (NE PAS LIBERER). */
extern "C" const double* ag_get_grad_ptr(int64_t id, int64_t* out_shape, int* out_ndim) {
    return ag_get_var_grad_ptr(id, out_shape, out_ndim);
}

// Operations : chacune prend des ids et retourne un id
extern "C" int64_t ag_relu(int64_t x_id) {
    auto v = ia_autograd::relu(ag_get_var(x_id));
    return v.id;
}

extern "C" int64_t ag_sigmoid(int64_t x_id) {
    auto v = ia_autograd::sigmoid(ag_get_var(x_id));
    return v.id;
}

extern "C" int64_t ag_tanh(int64_t x_id) {
    auto v = ia_autograd::tanh(ag_get_var(x_id));
    return v.id;
}

extern "C" int64_t ag_add(int64_t a_id, int64_t b_id) {
    auto v = ia_autograd::add(ag_get_var(a_id), ag_get_var(b_id));
    return v.id;
}

extern "C" int64_t ag_sub(int64_t a_id, int64_t b_id) {
    auto v = ia_autograd::sub(ag_get_var(a_id), ag_get_var(b_id));
    return v.id;
}

extern "C" int64_t ag_mul(int64_t a_id, int64_t b_id) {
    auto v = ia_autograd::mul(ag_get_var(a_id), ag_get_var(b_id));
    return v.id;
}

extern "C" int64_t ag_scale(int64_t x_id, double s) {
    auto v = ia_autograd::scale(ag_get_var(x_id), s);
    return v.id;
}

extern "C" int64_t ag_matmul(int64_t a_id, int64_t b_id) {
    auto v = ia_autograd::matmul(ag_get_var(a_id), ag_get_var(b_id));
    return v.id;
}

extern "C" int64_t ag_mse(int64_t pred_id, int64_t target_id) {
    auto v = ia_autograd::mse(ag_get_var(pred_id), ag_get_var(target_id));
    return v.id;
}

extern "C" int64_t ag_dot(int64_t a_id, int64_t b_id) {
    auto v = ia_autograd::dot(ag_get_var(a_id), ag_get_var(b_id));
    return v.id;
}

/** Declenche le backward depuis un Var. */
extern "C" void ag_backward(int64_t id) {
    ag_get_var(id).backward();
}