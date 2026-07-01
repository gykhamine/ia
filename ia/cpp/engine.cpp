/**
 * IA/cpp/engine.cpp — Implementation du moteur de calcul IA en C++ pur.
 *
 * Toutes les operations sont implementees sans numpy/pandas.
 * Utilise uniquement la bibliotheque standard C++ (CMATH, ALGORITHM, etc.).
 */

#include "engine.h"
#include <random>
#include <complex>
#include <functional>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Support OpenMP (Phase 2) : active si le compilateur a -fopenmp
// ============================================================================
#ifdef _OPENMP
#include <omp.h>
#define IA_HAS_OPENMP 1
#else
#define IA_HAS_OPENMP 0
#endif

namespace ia_core {

// ============================================================================
// Generateur de nombres aleatoires (thread-local pour securite)
// ============================================================================

static thread_local std::mt19937_64 rng_engine(42);

static void seed_rng(uint64_t seed) {
    rng_engine.seed(seed);
}

static double randn_scalar() {
    std::normal_distribution<double> dist(0.0, 1.0);
    return dist(rng_engine);
}

static double uniform_scalar(double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(rng_engine);
}

// ============================================================================
// 1. Tensor — constructeurs et methodes
// ============================================================================

Tensor::Tensor(Shape s) : data(s.size(), 0.0), shape(std::move(s)) {}

Tensor::Tensor(std::vector<double> d, Shape s)
    : data(std::move(d)), shape(std::move(s)) {}

double& Tensor::at(const std::vector<int64_t>& idx) {
    return data[ravel(idx)];
}

const double& Tensor::at(const std::vector<int64_t>& idx) const {
    return data[ravel(idx)];
}

int64_t Tensor::ravel(const std::vector<int64_t>& idx) const {
    int64_t linear = 0;
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.ndim()) - 1; i >= 0; --i) {
        linear += idx[i] * stride;
        stride *= shape[i];
    }
    return linear;
}

std::vector<int64_t> Tensor::unravel(int64_t linear) const {
    std::vector<int64_t> idx(shape.ndim());
    int64_t remaining = linear;
    for (int i = static_cast<int>(shape.ndim()) - 1; i >= 0; --i) {
        idx[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    return idx;
}

Tensor Tensor::reshape(Shape new_shape) const {
    int64_t new_size = new_shape.size();
    if (new_size != static_cast<int64_t>(data.size())) {
        throw std::runtime_error("reshape: size mismatch");
    }
    Tensor out;
    out.data = data;  // copie partagee du vecteur
    out.shape = new_shape;
    return out;
}

Tensor Tensor::flatten() const {
    return Tensor(data, Shape({static_cast<int64_t>(data.size())}));
}

Tensor Tensor::transpose2d() const {
    if (shape.ndim() != 2) {
        throw std::runtime_error("transpose2d: requires 2D tensor");
    }
    int64_t rows = shape[0], cols = shape[1];
    Tensor out(Shape({cols, rows}));
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            out.data[j * rows + i] = data[i * cols + j];
        }
    }
    return out;
}

Tensor Tensor::rows(int64_t start, int64_t end) const {
    if (shape.ndim() < 2) {
        throw std::runtime_error("rows: requires at least 2D tensor");
    }
    int64_t cols = 1;
    for (int i = 1; i < shape.ndim(); ++i) cols *= shape[i];
    int64_t len = end - start;
    std::vector<double> new_data(data.begin() + start * cols,
                                  data.begin() + end * cols);
    std::vector<int64_t> new_dims = shape.dims;
    new_dims[0] = len;
    return Tensor(std::move(new_data), Shape(new_dims));
}

Tensor Tensor::subtensor(int64_t offset, int64_t length) const {
    return Tensor(
        std::vector<double>(data.begin() + offset, data.begin() + offset + length),
        Shape({length})
    );
}

Tensor Tensor::copy() const {
    return Tensor(data, shape);
}

// ============================================================================
// 2. Fonctions d'activation
// ============================================================================

Tensor relu(const Tensor& x) {
    Tensor out = x.copy();
    for (auto& v : out.data) v = std::max(0.0, v);
    return out;
}

Tensor relu_deriv(const Tensor& x) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) {
        out.data[i] = x.data[i] > 0 ? 1.0 : 0.0;
    }
    return out;
}

Tensor sigmoid(const Tensor& x) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) {
        double v = std::max(-10.0, std::min(10.0, x.data[i]));
        out.data[i] = 1.0 / (1.0 + std::exp(-v));
    }
    return out;
}

Tensor sigmoid_deriv(const Tensor& x) {
    // sig(x) * (1 - sig(x))
    Tensor s = sigmoid(x);
    Tensor out(s.shape);
    for (size_t i = 0; i < s.data.size(); ++i) {
        out.data[i] = s.data[i] * (1.0 - s.data[i]);
    }
    return out;
}

Tensor tanh_act(const Tensor& x) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) {
        out.data[i] = std::tanh(x.data[i]);
    }
    return out;
}

Tensor tanh_deriv(const Tensor& x) {
    // 1 - tanh(x)^2
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) {
        double t = std::tanh(x.data[i]);
        out.data[i] = 1.0 - t * t;
    }
    return out;
}

Tensor leaky_relu(const Tensor& x, double alpha) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) {
        out.data[i] = x.data[i] > 0 ? x.data[i] : alpha * x.data[i];
    }
    return out;
}

Tensor leaky_relu_deriv(const Tensor& x, double alpha) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) {
        out.data[i] = x.data[i] > 0 ? 1.0 : alpha;
    }
    return out;
}

Tensor softmax(const Tensor& x) {
    // Softmax sur le dernier axe
    Tensor out(x.shape);
    int64_t last_dim = x.shape[x.ndim() - 1];
    int64_t n_slices = x.size() / last_dim;

    for (int64_t s = 0; s < n_slices; ++s) {
        int64_t base = s * last_dim;

        // Trouver le max pour stabilite numerique
        double max_val = x.data[base];
        for (int64_t j = 1; j < last_dim; ++j) {
            max_val = std::max(max_val, x.data[base + j]);
        }

        // exp(x - max)
        double sum_exp = 0.0;
        for (int64_t j = 0; j < last_dim; ++j) {
            out.data[base + j] = std::exp(x.data[base + j] - max_val);
            sum_exp += out.data[base + j];
        }

        // Normaliser
        for (int64_t j = 0; j < last_dim; ++j) {
            out.data[base + j] /= sum_exp;
        }
    }
    return out;
}

Tensor gelu(const Tensor& x) {
    Tensor out(x.shape);
    double c = std::sqrt(2.0 / M_PI);
    for (size_t i = 0; i < x.data.size(); ++i) {
        double v = x.data[i];
        double inner = c * (v + 0.044715 * v * v * v);
        out.data[i] = 0.5 * v * (1.0 + std::tanh(inner));
    }
    return out;
}

Tensor gelu_deriv(const Tensor& x) {
    Tensor out(x.shape);
    double c = std::sqrt(2.0 / M_PI);
    for (size_t i = 0; i < x.data.size(); ++i) {
        double v = x.data[i];
        double inner = c * (v + 0.044715 * v * v * v);
        double t = std::tanh(inner);
        double sech2 = 1.0 - t * t;
        out.data[i] = 0.5 * (1.0 + t) + 0.5 * v * sech2 * c * (1.0 + 3.0 * 0.044715 * v * v);
    }
    return out;
}

// ============================================================================
// 3. Algebre lineaire
// ============================================================================

/** Helper: multiplication de deux matrices 2D. */
static void matmul_2d(const double* A, int64_t M, int64_t K,
                      const double* B, int64_t K2, int64_t N,
                      double* C) {
    // K doit etre egal a K2
    // Phase 2 : parallelisation OpenMP sur la boucle externe (lignes de C).
    // Chaque thread ecrit dans une ligne disjointe de C -> pas de course.
    // L'inner loop sur K est garde scalaire pour laisser l'auto-vec du compilo
    // operer sur une boucle simple (souvent meilleur que #pragma omp simd ici).
#if IA_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < M; ++i) {
        const double* a_row = A + i * K;
        double* c_row = C + i * N;
        for (int64_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (int64_t p = 0; p < K; ++p) {
                s += a_row[p] * B[p * N + j];
            }
            c_row[j] = s;
        }
    }
}

Tensor matmul(const Tensor& A, const Tensor& B) {
    // Cas 1D x 2D : (K,) x (K, N) -> (N,)  (numpy broadcast)
    if (A.ndim() == 1 && B.ndim() == 2) {
        int64_t K = A.shape[0], K2 = B.shape[0], N = B.shape[1];
        if (K != K2) throw std::runtime_error("matmul: 1Dx2D shape mismatch");
        Tensor C(Shape({N}));
        matmul_2d(A.ptr(), 1, K, B.ptr(), K, N, C.ptr());
        return C;
    }

    // Cas 2D x 1D : (M, K) x (K,) -> (M,)  (numpy broadcast)
    if (A.ndim() == 2 && B.ndim() == 1) {
        int64_t M = A.shape[0], K = A.shape[1], K2 = B.shape[0];
        if (K != K2) throw std::runtime_error("matmul: 2Dx1D shape mismatch");
        Tensor C(Shape({M}));
        matmul_2d(A.ptr(), M, K, B.ptr(), K, 1, C.ptr());
        return C;
    }

    // Cas 2D x 2D
    if (A.ndim() == 2 && B.ndim() == 2) {
        int64_t M = A.shape[0], K = A.shape[1];
        int64_t K2 = B.shape[0], N = B.shape[1];
        if (K != K2) throw std::runtime_error("matmul: shape mismatch 2D");
        Tensor C(Shape({M, N}));
        matmul_2d(A.ptr(), M, K, B.ptr(), K, N, C.ptr());
        return C;
    }

    // Cas 3D batched (batch, M, K) x (batch, K, N)
    if (A.ndim() == 3 && B.ndim() == 3) {
        int64_t batch = A.shape[0];
        int64_t M = A.shape[1], K = A.shape[2];
        int64_t K2 = B.shape[1], N = B.shape[2];
        if (K != K2) throw std::runtime_error("matmul: shape mismatch 3D");
        Tensor C(Shape({batch, M, N}));
        for (int64_t b = 0; b < batch; ++b) {
            matmul_2d(A.ptr() + b * M * K, M, K,
                      B.ptr() + b * K * N, K, N,
                      C.ptr() + b * M * N);
        }
        return C;
    }

    // Cas 1D x 1D (dot product) -> scalaire
    if (A.ndim() == 1 && B.ndim() == 1) {
        if (A.size() != B.size()) throw std::runtime_error("matmul: 1D size mismatch");
        double s = 0.0;
        for (int64_t i = 0; i < A.size(); ++i) s += A[i] * B[i];
        return Tensor({s}, Shape({1}));
    }

    throw std::runtime_error("matmul: unsupported dimensions " +
        std::to_string(A.ndim()) + "x" + std::to_string(B.ndim()));
}

Tensor mul(const Tensor& A, const Tensor& B) {
    if (A.size() != B.size()) throw std::runtime_error("mul: size mismatch");
    Tensor out(A.shape);
    for (size_t i = 0; i < A.data.size(); ++i) {
        out.data[i] = A.data[i] * B.data[i];
    }
    return out;
}

Tensor add(const Tensor& A, const Tensor& B) {
    if (A.size() != B.size()) throw std::runtime_error("add: size mismatch");
    Tensor out(A.shape);
    for (size_t i = 0; i < A.data.size(); ++i) {
        out.data[i] = A.data[i] + B.data[i];
    }
    return out;
}

Tensor sub(const Tensor& A, const Tensor& B) {
    if (A.size() != B.size()) throw std::runtime_error("sub: size mismatch");
    Tensor out(A.shape);
    for (size_t i = 0; i < A.data.size(); ++i) {
        out.data[i] = A.data[i] - B.data[i];
    }
    return out;
}

Tensor div(const Tensor& A, const Tensor& B) {
    if (A.size() != B.size()) throw std::runtime_error("div: size mismatch");
    Tensor out(A.shape);
    for (size_t i = 0; i < A.data.size(); ++i) {
        out.data[i] = A.data[i] / (B.data[i] + 1e-12);
    }
    return out;
}

Tensor scale(const Tensor& x, double s) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) out.data[i] = x.data[i] * s;
    return out;
}

Tensor add_scalar(const Tensor& x, double s) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) out.data[i] = x.data[i] + s;
    return out;
}

Tensor clip(const Tensor& x, double lo, double hi) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) {
        out.data[i] = std::max(lo, std::min(hi, x.data[i]));
    }
    return out;
}

Tensor outer(const Tensor& a, const Tensor& B) {
    // a: (n,), B: (m,) -> (n, m) ou B: (m, k) -> (n, k)
    if (a.ndim() != 1) throw std::runtime_error("outer: first arg must be 1D");
    int64_t n = a.size();

    if (B.ndim() == 1) {
        int64_t m = B.size();
        Tensor out(Shape({n, m}));
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < m; ++j) {
                out.data[i * m + j] = a[i] * B[j];
            }
        }
        return out;
    }

    if (B.ndim() == 2) {
        int64_t m = B.shape[0], k = B.shape[1];
        Tensor out(Shape({n, k}));
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < m; ++j) {
                for (int64_t p = 0; p < k; ++p) {
                    out.data[i * k + p] += a[i] * B.data[j * k + p];
                }
            }
        }
        return out;
    }

    throw std::runtime_error("outer: unsupported B dimensions");
}

Tensor exp(const Tensor& x) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) out.data[i] = std::exp(x.data[i]);
    return out;
}

Tensor log(const Tensor& x) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) {
        out.data[i] = std::log(x.data[i] + 1e-12);
    }
    return out;
}

Tensor sqrt(const Tensor& x) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) {
        out.data[i] = std::sqrt(std::max(0.0, x.data[i]));
    }
    return out;
}

Tensor pow(const Tensor& x, double n) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) out.data[i] = std::pow(x.data[i], n);
    return out;
}

Tensor abs(const Tensor& x) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) out.data[i] = std::fabs(x.data[i]);
    return out;
}

Tensor sign(const Tensor& x) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) {
        if (x.data[i] > 0) out.data[i] = 1.0;
        else if (x.data[i] < 0) out.data[i] = -1.0;
        else out.data[i] = 0.0;
    }
    return out;
}

Tensor maximum(const Tensor& A, const Tensor& B) {
    if (A.size() != B.size()) throw std::runtime_error("maximum: size mismatch");
    Tensor out(A.shape);
    for (size_t i = 0; i < A.data.size(); ++i) {
        out.data[i] = std::max(A.data[i], B.data[i]);
    }
    return out;
}

Tensor neg(const Tensor& x) {
    Tensor out(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i) out.data[i] = -x.data[i];
    return out;
}

Tensor concatenate(const Tensor& A, const Tensor& B) {
    std::vector<double> merged = A.data;
    merged.insert(merged.end(), B.data.begin(), B.data.end());
    std::vector<int64_t> new_dims = A.shape.dims;
    new_dims[0] += B.shape[0];
    return Tensor(std::move(merged), Shape(new_dims));
}

Tensor tile(const Tensor& x, int64_t n) {
    int64_t row_size = x.size() / x.shape[0];
    std::vector<double> new_data;
    new_data.reserve(x.size() * n);
    for (int64_t t = 0; t < n; ++t) {
        new_data.insert(new_data.end(), x.data.begin(), x.data.end());
    }
    std::vector<int64_t> new_dims = x.shape.dims;
    new_dims[0] *= n;
    return Tensor(std::move(new_data), Shape(new_dims));
}

Tensor repeat(const Tensor& x, int64_t n) {
    std::vector<double> new_data;
    new_data.reserve(x.size() * n);
    for (auto v : x.data) {
        for (int64_t i = 0; i < n; ++i) new_data.push_back(v);
    }
    return Tensor(std::move(new_data), Shape({static_cast<int64_t>(new_data.size())}));
}

// ============================================================================
// 4. Convolutions
// ============================================================================

Tensor convolve2d(const Tensor& img, const Tensor& kernel) {
    int64_t ih = img.shape[0], iw = img.shape[1];
    int64_t kh = kernel.shape[0], kw = kernel.shape[1];
    int64_t oh = ih - kh + 1, ow = iw - kw + 1;
    Tensor out(Shape({oh, ow}));

    // Phase 2 : parallelisation sur les lignes de sortie (i).
    // Seuil : OpenMP seulement si la sortie est suffisamment grande pour
    // amortir le cout de demarrage des threads (~quelques us sur 4 coeurs).
#if IA_HAS_OPENMP
        #pragma omp parallel for if(oh * ow >= 65536) schedule(static) collapse(2)
#endif
    for (int64_t i = 0; i < oh; ++i) {
        for (int64_t j = 0; j < ow; ++j) {
            double s = 0.0;
            for (int64_t m = 0; m < kh; ++m) {
                for (int64_t n = 0; n < kw; ++n) {
                    s += img.data[(i + m) * iw + (j + n)] * kernel.data[m * kw + n];
                }
            }
            out.data[i * ow + j] = s;
        }
    }
    return out;
}

Tensor convolve2d_backward(const Tensor& img, const Tensor& kernel, const Tensor& d_conv) {
    int64_t kh = kernel.shape[0], kw = kernel.shape[1];
    int64_t ih = img.shape[0], iw = img.shape[1];
    int64_t oh = d_conv.shape[0], ow = d_conv.shape[1];
    Tensor d_kernel(Shape({kh, kw}));

    // Phase 2 : parallelisation sur les coefficients du kernel (m, n).
    // Seuil : OpenMP seulement si le kernel est suffisamment gros pour
    // amortir le cout de demarrage des threads.
#if IA_HAS_OPENMP
        #pragma omp parallel for if(kh * kw >= 64) schedule(static) collapse(2)
#endif
    for (int64_t m = 0; m < kh; ++m) {
        for (int64_t n = 0; n < kw; ++n) {
            double grad = 0.0;
            for (int64_t i = 0; i < oh; ++i) {
                for (int64_t j = 0; j < ow; ++j) {
                    if (i + m < ih && j + n < iw) {
                        grad += img.data[(i + m) * iw + (j + n)] * d_conv.data[i * ow + j];
                    }
                }
            }
            d_kernel.data[m * kw + n] = grad;
        }
    }
    return d_kernel;
}

Tensor conv1d_forward(const Tensor& x, const Tensor& W, const Tensor& b,
                      int kernel_size, int dilation) {
    int64_t in_c = W.shape[1];
    int64_t out_c = W.shape[0];
    int64_t seq_len = x.shape[1];
    int64_t eff_ks = (kernel_size - 1) * dilation + 1;
    int64_t out_len = std::max(int64_t(1), seq_len - eff_ks + 1);

    Tensor out(Shape({out_c, out_len}));

    for (int64_t oc = 0; oc < out_c; ++oc) {
        for (int64_t op = 0; op < out_len; ++op) {
            double val = 0.0;
            for (int64_t ic = 0; ic < in_c; ++ic) {
                for (int k = 0; k < kernel_size; ++k) {
                    int64_t idx = op + k * dilation;
                    if (idx < seq_len) {
                        val += x.data[ic * seq_len + idx] *
                               W.data[(oc * in_c + ic) * kernel_size + k];
                    }
                }
            }
            out.data[oc * out_len + op] = val + b.data[oc];
        }
    }
    return out;
}

std::vector<Tensor> conv1d_backward(const Tensor& x, const Tensor& W,
                                     const Tensor& d_out,
                                     int kernel_size, int dilation) {
    int64_t in_c = W.shape[1];
    int64_t out_c = W.shape[0];
    int64_t seq_len = x.shape[1];
    int64_t out_len = d_out.shape[1];

    Tensor d_W = zeros(W.shape);
    Tensor d_b = zeros(Shape({out_c}));
    Tensor d_x = zeros(x.shape);

    for (int64_t oc = 0; oc < out_c; ++oc) {
        for (int64_t op = 0; op < out_len; ++op) {
            for (int64_t ic = 0; ic < in_c; ++ic) {
                for (int k = 0; k < kernel_size; ++k) {
                    int64_t idx = op + k * dilation;
                    if (idx < seq_len) {
                        int64_t w_idx = (oc * in_c + ic) * kernel_size + k;
                        d_W.data[w_idx] += x.data[ic * seq_len + idx] * d_out.data[oc * out_len + op];
                        d_x.data[ic * seq_len + idx] += W.data[w_idx] * d_out.data[oc * out_len + op];
                    }
                }
            }
            d_b.data[oc] += d_out.data[oc * out_len + op];
        }
    }
    return {d_x, d_W, d_b};
}

/** Helper recursif pour convolve_nd. */
static void conv_nd_helper(const double* vol, const double* ker,
                            const std::vector<int64_t>& v_shape,
                            const std::vector<int64_t>& k_shape,
                            std::vector<int64_t>& out_shape,
                            std::vector<int64_t>& idx,
                            int dim, double& accum) {
    if (dim == static_cast<int>(v_shape.size())) {
        // Calculer la somme volume[slice] * kernel
        double s = 0.0;
        std::vector<int64_t> v_idx(v_shape.size());
        for (int64_t d = 0; d < static_cast<int64_t>(v_shape.size()); ++d) {
            v_idx[d] = idx[d];
        }
        // Iterer sur le kernel
        std::vector<int64_t> k_idx(v_shape.size(), 0);
        int64_t total = 1;
        for (auto d : k_shape) total *= d;

        // Flat iteration over kernel
        for (int64_t ki = 0; ki < total; ++ki) {
            // Decomposer ki en indices kernel
            int64_t rem = ki;
            bool in_bounds = true;
            for (int d = static_cast<int>(k_shape.size()) - 1; d >= 0; --d) {
                k_idx[d] = rem % k_shape[d];
                rem /= k_shape[d];
                if (idx[d] + k_idx[d] >= v_shape[d]) {
                    in_bounds = false;
                    break;
                }
            }
            if (!in_bounds) continue;

            // Calculer l'index dans le volume
            int64_t v_linear = 0;
            int64_t stride = 1;
            for (int d = static_cast<int>(v_shape.size()) - 1; d >= 0; --d) {
                v_linear += (idx[d] + k_idx[d]) * stride;
                stride *= v_shape[d];
            }
            s += vol[v_linear] * ker[ki];
        }
        accum = s;
        return;
    }

    for (int64_t i = 0; i < out_shape[dim]; ++i) {
        idx[dim] = i;
        conv_nd_helper(vol, ker, v_shape, k_shape, out_shape, idx, dim + 1, accum);
    }
}

Tensor convolve_nd(const Tensor& volume, const Tensor& kernel) {
    int64_t nd = volume.ndim();
    std::vector<int64_t> v_shape(volume.shape.dims.begin(), volume.shape.dims.end());
    std::vector<int64_t> k_shape(kernel.shape.dims.begin(), kernel.shape.dims.end());
    std::vector<int64_t> out_shape(nd);
    int64_t out_size = 1;
    for (int i = 0; i < nd; ++i) {
        out_shape[i] = v_shape[i] - k_shape[i] + 1;
        out_size *= out_shape[i];
    }

    Tensor out = Tensor(Shape(out_shape));
    std::vector<int64_t> idx(nd, 0);

    // Iteration flat sur l'espace de sortie
    for (int64_t oi = 0; oi < out_size; ++oi) {
        // Decomposer oi
        int64_t rem = oi;
        for (int d = nd - 1; d >= 0; --d) {
            idx[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }

        double s = 0.0;
        int64_t k_total = 1;
        for (auto d : k_shape) k_total *= d;

        for (int64_t ki = 0; ki < k_total; ++ki) {
            int64_t krem = ki;
            std::vector<int64_t> k_idx(nd);
            bool in_bounds = true;
            for (int d = nd - 1; d >= 0; --d) {
                k_idx[d] = krem % k_shape[d];
                krem /= k_shape[d];
                if (idx[d] + k_idx[d] >= v_shape[d]) { in_bounds = false; break; }
            }
            if (!in_bounds) continue;

            int64_t v_linear = 0;
            int64_t stride = 1;
            for (int d = nd - 1; d >= 0; --d) {
                v_linear += (idx[d] + k_idx[d]) * stride;
                stride *= v_shape[d];
            }
            s += volume.data[v_linear] * kernel.data[ki];
        }
        out.data[oi] = s;
    }
    return out;
}

Tensor convolve_nd_backward(const Tensor& volume, const Tensor& kernel, const Tensor& d_conv) {
    int64_t nd = volume.ndim();
    auto v_shape = volume.shape.dims;
    auto k_shape = kernel.shape.dims;
    auto d_shape = d_conv.shape.dims;
    int64_t k_total = 1;
    for (auto d : k_shape) k_total *= d;

    Tensor d_kernel = zeros(kernel.shape);

    for (int64_t ki = 0; ki < k_total; ++ki) {
        int64_t krem = ki;
        std::vector<int64_t> k_idx(nd);
        for (int d = nd - 1; d >= 0; --d) {
            k_idx[d] = krem % k_shape[d];
            krem /= k_shape[d];
        }

        double grad = 0.0;
        int64_t d_total = 1;
        for (auto d : d_shape) d_total *= d;

        for (int64_t di = 0; di < d_total; ++di) {
            int64_t drem = di;
            std::vector<int64_t> out_idx(nd);
            for (int d = nd - 1; d >= 0; --d) {
                out_idx[d] = drem % d_shape[d];
                drem /= d_shape[d];
            }

            bool in_bounds = true;
            for (int d = 0; d < nd; ++d) {
                if (out_idx[d] + k_idx[d] >= v_shape[d]) { in_bounds = false; break; }
            }
            if (!in_bounds) continue;

            int64_t v_linear = 0;
            int64_t stride = 1;
            for (int d = nd - 1; d >= 0; --d) {
                v_linear += (out_idx[d] + k_idx[d]) * stride;
                stride *= v_shape[d];
            }
            grad += volume.data[v_linear] * d_conv.data[di];
        }
        d_kernel.data[ki] = grad;
    }
    return d_kernel;
}

// ============================================================================
// 5. Normalisation
// ============================================================================

Tensor layer_norm(const Tensor& x, double eps) {
    Tensor out(x.shape);
    int64_t last_dim = x.shape[x.ndim() - 1];
    int64_t n_slices = x.size() / last_dim;

    for (int64_t s = 0; s < n_slices; ++s) {
        int64_t base = s * last_dim;

        // Moyenne
        double mean = 0.0;
        for (int64_t j = 0; j < last_dim; ++j) mean += x.data[base + j];
        mean /= last_dim;

        // Variance
        double var = 0.0;
        for (int64_t j = 0; j < last_dim; ++j) {
            double d = x.data[base + j] - mean;
            var += d * d;
        }
        var /= last_dim;

        double inv_std = 1.0 / std::sqrt(var + eps);
        for (int64_t j = 0; j < last_dim; ++j) {
            out.data[base + j] = (x.data[base + j] - mean) * inv_std;
        }
    }
    return out;
}

// ============================================================================
// 6. Fonctions de perte
// ============================================================================

double mse_loss(const Tensor& pred, const Tensor& target) {
    double s = 0.0;
    int64_t n = pred.size();
    for (int64_t i = 0; i < n; ++i) {
        double d = pred.data[i] - target.data[i];
        s += d * d;
    }
    return s / n;
}

Tensor mse_loss_grad(const Tensor& pred, const Tensor& target) {
    int64_t n = pred.size();
    Tensor out(pred.shape);
    for (int64_t i = 0; i < n; ++i) {
        out.data[i] = 2.0 * (pred.data[i] - target.data[i]) / n;
    }
    return out;
}

double cross_entropy_loss(const Tensor& logits, int target_idx) {
    // softmax puis -log(soft[target_idx])
    int64_t n = logits.size();
    double max_val = logits.data[0];
    for (int64_t i = 1; i < n; ++i) max_val = std::max(max_val, logits.data[i]);

    double sum_exp = 0.0;
    std::vector<double> exp_vals(n);
    for (int64_t i = 0; i < n; ++i) {
        exp_vals[i] = std::exp(logits.data[i] - max_val);
        sum_exp += exp_vals[i];
    }
    return -std::log(exp_vals[target_idx] / sum_exp + 1e-12);
}

// ============================================================================
// 7. Initialisation
// ============================================================================

Tensor zeros(Shape shape) {
    return Tensor(shape);
}

Tensor ones(Shape shape) {
    Tensor t(shape);
    std::fill(t.data.begin(), t.data.end(), 1.0);
    return t;
}

Tensor xavier_init(Shape shape, uint64_t seed) {
    seed_rng(seed);
    int64_t fan_sum = 0;
    for (auto d : shape.dims) fan_sum += d;
    double std_dev = std::sqrt(2.0 / fan_sum);

    Tensor t(shape);
    for (auto& v : t.data) v = randn_scalar() * std_dev;
    return t;
}

Tensor randn(Shape shape, uint64_t seed) {
    seed_rng(seed);
    Tensor t(shape);
    for (auto& v : t.data) v = randn_scalar();
    return t;
}

Tensor uniform(Shape shape, double lo, double hi, uint64_t seed) {
    seed_rng(seed);
    Tensor t(shape);
    for (auto& v : t.data) v = uniform_scalar(lo, hi);
    return t;
}

std::vector<int64_t> permutation(int64_t n, uint64_t seed) {
    seed_rng(seed);
    std::vector<int64_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    // Fisher-Yates
    for (int64_t i = n - 1; i > 0; --i) {
        int64_t j = static_cast<int64_t>(uniform_scalar(0, i + 1));
        std::swap(idx[i], idx[j]);
    }
    return idx;
}

int64_t randint(int64_t lo, int64_t hi, uint64_t& seed) {
    std::uniform_int_distribution<int64_t> dist(lo, hi - 1);
    return dist(rng_engine);
}

// ============================================================================
// 8. Operations de reduction
// ============================================================================

double sum(const Tensor& x) {
    return std::accumulate(x.data.begin(), x.data.end(), 0.0);
}

Tensor sum_axis(const Tensor& x, int axis) {
    if (axis == -1 || axis == static_cast<int>(x.ndim()) - 1) {
        // Somme sur le dernier axe
        int64_t last = x.shape[x.ndim() - 1];
        int64_t n_slices = x.size() / last;
        Tensor out(Shape({n_slices}));
        for (int64_t s = 0; s < n_slices; ++s) {
            double s_val = 0.0;
            for (int64_t j = 0; j < last; ++j) {
                s_val += x.data[s * last + j];
            }
            out.data[s] = s_val;
        }
        return out;
    }
    if (axis == 0) {
        int64_t first = x.shape[0];
        int64_t rest = x.size() / first;
        Tensor out(Shape({rest}));
        for (int64_t j = 0; j < rest; ++j) {
            double s_val = 0.0;
            for (int64_t i = 0; i < first; ++i) {
                s_val += x.data[i * rest + j];
            }
            out.data[j] = s_val;
        }
        return out;
    }
    throw std::runtime_error("sum_axis: unsupported axis");
}

double mean(const Tensor& x) {
    return sum(x) / x.size();
}

Tensor mean_axis(const Tensor& x, int axis) {
    if (axis == -1 || axis == static_cast<int>(x.ndim()) - 1) {
        int64_t last = x.shape[x.ndim() - 1];
        Tensor s = sum_axis(x, axis);
        return scale(s, 1.0 / last);
    }
    if (axis == 0) {
        int64_t first = x.shape[0];
        Tensor s = sum_axis(x, 0);
        return scale(s, 1.0 / first);
    }
    if (axis == 1) {
        // Pour 2D: (rows, cols) -> (rows,)
        if (x.ndim() != 2) throw std::runtime_error("mean_axis(1): requires 2D");
        int64_t rows = x.shape[0], cols = x.shape[1];
        Tensor out(Shape({rows}));
        for (int64_t i = 0; i < rows; ++i) {
            double s = 0.0;
            for (int64_t j = 0; j < cols; ++j) s += x.data[i * cols + j];
            out.data[i] = s / cols;
        }
        return out;
    }
    throw std::runtime_error("mean_axis: unsupported axis");
}

Tensor var_axis(const Tensor& x, int axis) {
    int64_t last = x.shape[axis == -1 ? x.ndim() - 1 : axis];
    Tensor m;
    if (axis == -1 || axis == static_cast<int>(x.ndim()) - 1) {
        m = mean_axis(x, -1);
        int64_t n_slices = x.size() / last;
        Tensor v(Shape({n_slices, last}));
        for (int64_t s = 0; s < n_slices; ++s) {
            for (int64_t j = 0; j < last; ++j) {
                double d = x.data[s * last + j] - m.data[s];
                v.data[s * last + j] = d * d;
            }
        }
        return mean_axis(v, -1);
    }
    throw std::runtime_error("var_axis: unsupported axis");
}

double max_val(const Tensor& x) {
    return *std::max_element(x.data.begin(), x.data.end());
}

Tensor max_axis(const Tensor& x) {
    // Max sur le dernier axe
    int64_t last = x.shape[x.ndim() - 1];
    int64_t n_slices = x.size() / last;
    Tensor out(Shape({n_slices}));
    for (int64_t s = 0; s < n_slices; ++s) {
        double m = x.data[s * last];
        for (int64_t j = 1; j < last; ++j) {
            m = std::max(m, x.data[s * last + j]);
        }
        out.data[s] = m;
    }
    return out;
}

int64_t argmax(const Tensor& x) {
    int64_t idx = 0;
    double best = x.data[0];
    for (size_t i = 1; i < x.data.size(); ++i) {
        if (x.data[i] > best) { best = x.data[i]; idx = static_cast<int64_t>(i); }
    }
    return idx;
}

std::vector<double> histogram(const Tensor& x, int n_bins, double range_lo, double range_hi) {
    std::vector<double> counts(n_bins, 0.0);
    double bin_width = (range_hi - range_lo) / n_bins;
    for (auto v : x.data) {
        int b = static_cast<int>((v - range_lo) / bin_width);
        if (b >= 0 && b < n_bins) counts[b] += 1.0;
    }
    return counts;
}

// ============================================================================
// 9. FFT simplifiee
// ============================================================================

Tensor fft_rfft(const Tensor& x, int n_fft) {
    int64_t n = x.size();
    int N = n_fft;
    int half = N / 2 + 1;

    // Zero-pad si necessaire
    std::vector<std::complex<double>> X(N, 0.0);
    for (int64_t i = 0; i < n; ++i) X[i] = x.data[i];

    // FFT radix-2 Cooley-Tukey
    int logN = 0;
    while ((1 << logN) < N) logN++;

    // Bit-reversal permutation
    for (int i = 0; i < N; ++i) {
        int j = 0;
        int tmp = i;
        for (int b = 0; b < logN; ++b) {
            j = (j << 1) | (tmp & 1);
            tmp >>= 1;
        }
        if (j > i) std::swap(X[i], X[j]);
    }

    // Butterfly
    for (int s = 1; s <= logN; ++s) {
        int m = 1 << s;
        std::complex<double> wm = std::exp(std::complex<double>(0, -2.0 * M_PI / m));
        for (int k = 0; k < N; k += m) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < m / 2; ++j) {
                auto t = w * X[k + j + m / 2];
                auto u = X[k + j];
                X[k + j] = u + t;
                X[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }

    // Retourner les magnitudes (half+1 bins)
    Tensor out(Shape({half}));
    for (int i = 0; i < half; ++i) {
        out.data[i] = std::abs(X[i]);
    }
    return out;
}

// ============================================================================
// 10. Operations specifiques
// ============================================================================

Tensor add_noise(const Tensor& x0, const Tensor& sqrt_alpha,
                 const Tensor& sqrt_one_minus) {
    Tensor noise = randn(x0.shape, 0);  // seed temporaire
    Tensor out(x0.shape);
    for (size_t i = 0; i < x0.size(); ++i) {
        out.data[i] = sqrt_alpha[0] * x0.data[i] + sqrt_one_minus[0] * noise.data[i];
    }
    return out;
}

Tensor linear(const Tensor& x, const Tensor& W, const Tensor& b) {
    return add(matmul(x, W), b);
}

double dot1d(const Tensor& a, const Tensor& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) s += a.data[i] * b.data[i];
    return s;
}

LDMForwardResult ldm_predict_noise(const Tensor& x_noisy,
                                    const Tensor& class_embedding, int class_id,
                                    const Tensor& W1, const Tensor& b1,
                                    const Tensor& W2, const Tensor& b2) {
    int64_t embed_dim = class_embedding.shape[1];

    // Extraire l'embedding de classe
    std::vector<double> c_embed_data(embed_dim);
    for (int64_t j = 0; j < embed_dim; ++j) {
        c_embed_data[j] = class_embedding.data[class_id * embed_dim + j];
    }
    Tensor c_embed(std::move(c_embed_data), Shape({embed_dim}));

    // Concatener
    Tensor x_concat = concatenate(x_noisy, c_embed);

    // FC1
    Tensor z1 = add(matmul(x_concat, W1), b1);
    Tensor h1 = relu(z1);

    // FC2
    Tensor output = add(matmul(h1, W2), b2);

    return {output, x_concat, z1, h1};
}

Tensor pad1d(const Tensor& x, int64_t target_len) {
    if (x.size() >= target_len) return x.copy();
    std::vector<double> padded = x.data;
    padded.resize(target_len, 0.0);
    return Tensor(std::move(padded), Shape({target_len}));
}

Tensor diff_axis(const Tensor& x, int axis) {
    if (axis == 1 && x.ndim() == 2) {
        int64_t rows = x.shape[0], cols = x.shape[1];
        Tensor out(Shape({rows, cols - 1}));
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols - 1; ++j) {
                out.data[i * (cols - 1) + j] = x.data[i * cols + j + 1] - x.data[i * cols + j];
            }
        }
        return out;
    }
    if (axis == 0 && x.ndim() == 2) {
        int64_t rows = x.shape[0], cols = x.shape[1];
        Tensor out(Shape({rows - 1, cols}));
        for (int64_t i = 0; i < rows - 1; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                out.data[i * cols + j] = x.data[(i + 1) * cols + j] - x.data[i * cols + j];
            }
        }
        return out;
    }
    // 1D diff
    if (x.ndim() == 1) {
        int64_t n = x.size();
        Tensor out(Shape({n - 1}));
        for (int64_t i = 0; i < n - 1; ++i) out.data[i] = x.data[i + 1] - x.data[i];
        return out;
    }
    throw std::runtime_error("diff_axis: unsupported");
}

Tensor linspace(double start, double end, int64_t num) {
    std::vector<double> data(num);
    if (num == 1) { data[0] = start; return Tensor(std::move(data), Shape({num})); }
    double step = (end - start) / (num - 1);
    for (int64_t i = 0; i < num; ++i) data[i] = start + i * step;
    return Tensor(std::move(data), Shape({num}));
}

Tensor arange(double start, double end, double step) {
    std::vector<double> data;
    for (double v = start; v < end; v += step) data.push_back(v);
    return Tensor(std::move(data), Shape({static_cast<int64_t>(data.size())}));
}

std::vector<Tensor> mgrid3d(int size) {
    int total = size * size * size;
    std::vector<double> gx(total), gy(total), gz(total);
    int idx = 0;
    for (int z = 0; z < size; ++z)
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x) {
                gx[idx] = x; gy[idx] = y; gz[idx] = z;
                ++idx;
            }
    return {
        Tensor(std::move(gx), Shape({size, size, size})),
        Tensor(std::move(gy), Shape({size, size, size})),
        Tensor(std::move(gz), Shape({size, size, size}))
    };
}

Tensor gather(const Tensor& x, const std::vector<int64_t>& indices) {
    std::vector<double> out(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) out[i] = x.data[indices[i]];
    return Tensor(std::move(out), Shape({static_cast<int64_t>(indices.size())}));
}

void scatter(Tensor& x, int64_t idx, const Tensor& value) {
    for (size_t i = 0; i < value.data.size() && idx + i < x.size(); ++i) {
        x.data[idx + i] = value.data[i];
    }
}

Tensor dct2(const Tensor& x, int num_coeffs) {
    int64_t n = x.size();
    int64_t K = std::min(num_coeffs, static_cast<int>(n));
    Tensor out(Shape({K}));
    for (int64_t k = 0; k < K; ++k) {
        double s = 0.0;
        for (int64_t n_idx = 0; n_idx < n; ++n_idx) {
            s += x.data[n_idx] * std::cos(M_PI * k * (2 * n_idx + 1) / (2 * n));
        }
        out.data[k] = s;
    }
    return out;
}

// ============================================================================
// 11. API bas niveau "zero-copy" (Phase 1)
// ============================================================================

void relu_inplace(double* x, int64_t n) {
    // Boucle simple, l'auto-vec de -O3 -march=native genère du SIMD automatique
    // (souvent AVX2 sur x86_64 moderne -> 4 doubles par iteration).
#if IA_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        if (x[i] < 0.0) x[i] = 0.0;
    }
}

void sigmoid_inplace(double* x, int64_t n) {
#if IA_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        double v = x[i];
        if (v >  10.0) v =  10.0;
        if (v < -10.0) v = -10.0;
        x[i] = 1.0 / (1.0 + std::exp(-v));
    }
}

void tanh_inplace(double* x, int64_t n) {
#if IA_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        x[i] = std::tanh(x[i]);
    }
}

void add_inplace(double* a, const double* b, int64_t n) {
#if IA_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) a[i] += b[i];
}

void axpy_inplace(double* a, const double* b, int64_t n, double s) {
#if IA_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) a[i] += s * b[i];
}

void scale_inplace(double* a, int64_t n, double s) {
#if IA_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) a[i] *= s;
}

void matmul_2d_into(const double* A, int64_t M, int64_t K,
                    const double* B, int64_t N,
                    double* C) {
    // Reutilise exactement la meme boucle que matmul_2d (avec OpenMP),
    // mais sans creer de Tensor. Le buffer C doit etre pre-alloue (M*N doubles).
    matmul_2d(A, M, K, B, K, N, C);
}

void convolve2d_into(const double* img, int64_t ih, int64_t iw,
                     const double* kernel, int64_t kh, int64_t kw,
                     double* out) {
    int64_t oh = ih - kh + 1, ow = iw - kw + 1;
#if IA_HAS_OPENMP
        #pragma omp parallel for if(oh * ow >= 65536) schedule(static) collapse(2)
#endif
    for (int64_t i = 0; i < oh; ++i) {
        for (int64_t j = 0; j < ow; ++j) {
            double s = 0.0;
            for (int64_t m = 0; m < kh; ++m) {
                for (int64_t n = 0; n < kw; ++n) {
                    s += img[(i + m) * iw + (j + n)] * kernel[m * kw + n];
                }
            }
            out[i * ow + j] = s;
        }
    }
}

} // namespace ia_core