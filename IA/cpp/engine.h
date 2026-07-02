/**
 * IA/cpp/engine.h — Moteur de calcul IA en C++ pur.
 *
 * Toutes les opérations fondamentales utilisées par les modules
 * train/ et infer/ sont implémentées ici, sans aucune dépendance
 * à numpy ou pandas. Les tableaux sont manipulés via des pointeurs
 * bruts (double*) pour des performances maximales.
 *
 * Sections :
 *   1. Utilitaires bas niveau (allocation, copie, reshape, flatten)
 *   2. Fonctions d'activation et leurs dérivées
 *   3. Opérations d'algebre lineaire (matmul, outer, transpose...)
 *   4. Convolutions (2D, ND, 1D avec dilation)
 *   5. Normalisation (layer_norm)
 *   6. Fonctions de perte (MSE, cross-entropy)
 *   7. Initialisation (xavier, randn, zeros)
 *   8. Operations de reduction (mean, sum, var, max, argmax)
 */

#ifndef IA_ENGINE_H
#define IA_ENGINE_H

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace ia_core {

// ============================================================================
// 1. Types et utilitaires bas niveau
// ============================================================================

/** Structure legere pour representer la forme d'un tableau ND. */
struct Shape {
    std::vector<int64_t> dims;

    Shape() = default;
    Shape(std::vector<int64_t> d) : dims(std::move(d)) {}
    explicit Shape(std::initializer_list<int64_t> il) : dims(il) {}

    int64_t size() const {
        if (dims.empty()) return 0;
        int64_t s = 1;
        for (auto d : dims) s *= d;
        return s;
    }

    int64_t ndim() const { return static_cast<int64_t>(dims.size()); }

    int64_t operator[](int i) const { return dims[i]; }
};

/** Classe Tensor minimaliste : stocke un tableau 1D de doubles avec sa forme. */
class Tensor {
public:
    std::vector<double> data;
    Shape shape;

    Tensor() = default;
    Tensor(Shape s);
    Tensor(std::vector<double> d, Shape s);

    double& operator[](int64_t i) { return data[i]; }
    const double& operator[](int64_t i) const { return data[i]; }

    int64_t size() const { return static_cast<int64_t>(data.size()); }
    int64_t ndim() const { return shape.ndim(); }

    /** Acces multi-index (row-major). */
    double& at(const std::vector<int64_t>& idx);
    const double& at(const std::vector<int64_t>& idx) const;

    /** Convertir un index lineaire en indices multi-dim. */
    std::vector<int64_t> unravel(int64_t linear) const;

    /** Convertir des indices multi-dim en index lineaire. */
    int64_t ravel(const std::vector<int64_t>& idx) const;

    /** Reshape (ne copie pas les donnees, verifie la taille). */
    Tensor reshape(Shape new_shape) const;

    /** Flatten en 1D. */
    Tensor flatten() const;

    /** Transpose 2D. */
    Tensor transpose2d() const;

    /** Slice sur l'axe 0 : retourne les lignes [start, end). */
    Tensor rows(int64_t start, int64_t end) const;

    /** Extrait une sous-Tensor continue. */
    Tensor subtensor(int64_t offset, int64_t length) const;

    /** Copie profonde. */
    Tensor copy() const;

    /** Pointeur brut vers les donnees. */
    double* ptr() { return data.data(); }
    const double* ptr() const { return data.data(); }
};

// ============================================================================
// 2. Fonctions d'activation
// ============================================================================

Tensor relu(const Tensor& x);
Tensor relu_deriv(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor sigmoid_deriv(const Tensor& x);      // sig(x) * (1 - sig(x))
Tensor tanh_act(const Tensor& x);           // evite conflit avec std::tanh
Tensor tanh_deriv(const Tensor& x);         // 1 - tanh(x)^2
Tensor leaky_relu(const Tensor& x, double alpha = 0.01);
Tensor leaky_relu_deriv(const Tensor& x, double alpha = 0.01);
Tensor softmax(const Tensor& x);            // sur le dernier axe
Tensor gelu(const Tensor& x);
Tensor gelu_deriv(const Tensor& x);

// ============================================================================
// 3. Algebre lineaire
// ============================================================================

/** Multiplication matricielle C = A @ B. Supporte 2D et 3D (batched). */
Tensor matmul(const Tensor& A, const Tensor& B);

/** Produit scalaire element-wise A * B (broadcasting 1D vs 2D basique). */
Tensor mul(const Tensor& A, const Tensor& B);

/** Addition element-wise A + B. */
Tensor add(const Tensor& A, const Tensor& B);

/** Soustraction element-wise A - B. */
Tensor sub(const Tensor& A, const Tensor& B);

/** Produit exterieur : outer(a, b) ou outer(a, B) si B est 2D. */
Tensor outer(const Tensor& a, const Tensor& B);

/** Division element-wise A / B. */
Tensor div(const Tensor& A, const Tensor& B);

/** Scalaire * Tensor. */
Tensor scale(const Tensor& x, double s);

/** Tensor + scalaire. */
Tensor add_scalar(const Tensor& x, double s);

/** Clip les valeurs entre lo et hi (in-place sur une copie). */
Tensor clip(const Tensor& x, double lo, double hi);

/** Exp element-wise. */
Tensor exp(const Tensor& x);

/** Log element-wise (log naturel). */
Tensor log(const Tensor& x);

/** Sqrt element-wise. */
Tensor sqrt(const Tensor& x);

/** Puissance element-wise x^n. */
Tensor pow(const Tensor& x, double n);

/** Abs element-wise. */
Tensor abs(const Tensor& x);

/** Signe element-wise. */
Tensor sign(const Tensor& x);

/** Maximum element-wise entre A et B. */
Tensor maximum(const Tensor& A, const Tensor& B);

/** Negation element-wise. */
Tensor neg(const Tensor& x);

/** Concatene deux Tensors sur l'axe 0. */
Tensor concatenate(const Tensor& A, const Tensor& B);

/** Tile: repete le Tensor sur l'axe 0 n fois. */
Tensor tile(const Tensor& x, int64_t n);

/** Repeat: repete chaque element n fois (comme np.repeat). */
Tensor repeat(const Tensor& x, int64_t n);

// ============================================================================
// 4. Convolutions
// ============================================================================

/** Convolution 2D : (ih, iw) x (kh, kw) -> (oh, ow). */
Tensor convolve2d(const Tensor& img, const Tensor& kernel);

/** Retropropagation convolution 2D : gradient du kernel. */
Tensor convolve2d_backward(const Tensor& img, const Tensor& kernel, const Tensor& d_conv);

/** Convolution 1D avec dilation.
 *  x : (in_channels, seq_len)
 *  W : (out_channels, in_channels, kernel_size)
 *  b : (out_channels,)
 *  Retourne (out_channels, out_seq_len).
 */
Tensor conv1d_forward(const Tensor& x, const Tensor& W, const Tensor& b,
                      int kernel_size, int dilation);

/** Retropropagation convolution 1D. Retourne (d_x, d_W, d_b). */
std::vector<Tensor> conv1d_backward(const Tensor& x, const Tensor& W,
                                     const Tensor& d_out,
                                     int kernel_size, int dilation);

/** Convolution ND generique.
 *  volume et kernel sont des Tensors avec shape.ndim() dimensions.
 *  Utilise la recursion pour iterer sur tous les indices de sortie.
 */
Tensor convolve_nd(const Tensor& volume, const Tensor& kernel);

/** Retropropagation convolution ND : gradient du kernel. */
Tensor convolve_nd_backward(const Tensor& volume, const Tensor& kernel, const Tensor& d_conv);

// ============================================================================
// 5. Normalisation
// ============================================================================

/** Layer normalization sur le dernier axe. */
Tensor layer_norm(const Tensor& x, double eps = 1e-8);

// ============================================================================
// 6. Fonctions de perte
// ============================================================================

/** MSE : mean((pred - target)^2). */
double mse_loss(const Tensor& pred, const Tensor& target);

/** Gradient MSE par rapport a pred : 2*(pred - target) / n. */
Tensor mse_loss_grad(const Tensor& pred, const Tensor& target);

/** Cross-entropy : -log(soft[pred][target_idx]). */
double cross_entropy_loss(const Tensor& logits, int target_idx);

// ============================================================================
// 7. Initialisation
// ============================================================================

/** Tensor rempli de zeros. */
Tensor zeros(Shape shape);

/** Tensor rempli de uns. */
Tensor ones(Shape shape);

/** Xavier/Glorot initialisation : randn * sqrt(2 / sum(shape)). */
Tensor xavier_init(Shape shape, uint64_t seed = 42);

/** Distribution normale N(0,1) avec seed. */
Tensor randn(Shape shape, uint64_t seed = 42);

/** Distribution uniforme U(lo, hi) avec seed. */
Tensor uniform(Shape shape, double lo, double hi, uint64_t seed = 42);

/** Permutation aleatoire d'indices [0..n-1]. */
std::vector<int64_t> permutation(int64_t n, uint64_t seed = 42);

/** Entier aleatoire dans [lo, hi). */
int64_t randint(int64_t lo, int64_t hi, uint64_t& seed);

// ============================================================================
// 8. Operations de reduction
// ============================================================================

/** Somme de tous les elements. */
double sum(const Tensor& x);

/** Somme sur un axe (0 ou -1). */
Tensor sum_axis(const Tensor& x, int axis);

/** Moyenne de tous les elements. */
double mean(const Tensor& x);

/** Moyenne sur un axe. */
Tensor mean_axis(const Tensor& x, int axis);

/** Variance sur le dernier axe. */
Tensor var_axis(const Tensor& x, int axis);

/** Maximum de tous les elements. */
double max_val(const Tensor& x);

/** Maximum sur le dernier axe. */
Tensor max_axis(const Tensor& x);

/** Argmax sur le dernier axe. */
int64_t argmax(const Tensor& x);

/** Histogramme : compte les valeurs dans [range_lo, range_hi] sur n_bins. */
std::vector<double> histogram(const Tensor& x, int n_bins, double range_lo, double range_hi);

// ============================================================================
// 9. Operations FFT simplifiees (pour audio/vision)
// ============================================================================

/** FFT 1D (Cooley-Tukey radix-2). n doit etre une puissance de 2. */
Tensor fft_rfft(const Tensor& x, int n_fft);

// ============================================================================
// 10. Operations specifiques aux modeles
// ============================================================================

/** Ajout de bruit de diffusion : sqrt(alpha) * x0 + sqrt(1-alpha) * noise. */
Tensor add_noise(const Tensor& x0, const Tensor& sqrt_alpha,
                 const Tensor& sqrt_one_minus);

/** Lineaire : x @ W + b. */
Tensor linear(const Tensor& x, const Tensor& W, const Tensor& b);

/** Dot product entre deux vecteurs 1D. */
double dot1d(const Tensor& a, const Tensor& b);

/** Predict noise pour LDM : concat(x_noisy, class_emb) @ W1 + b1 -> relu -> @ W2 + b2. */
struct LDMForwardResult {
    Tensor output;
    Tensor x_concat;
    Tensor z1;
    Tensor h1;
};
LDMForwardResult ldm_predict_noise(const Tensor& x_noisy,
                                    const Tensor& class_embedding, int class_id,
                                    const Tensor& W1, const Tensor& b1,
                                    const Tensor& W2, const Tensor& b2);

/** Pad 1D a la taille target. */
Tensor pad1d(const Tensor& x, int64_t target_len);

/** Diff absolu sur un axe. */
Tensor diff_axis(const Tensor& x, int axis);

/** Linspace comme np.linspace. */
Tensor linspace(double start, double end, int64_t num);

/** Arange comme np.arange. */
Tensor arange(double start, double end, double step);

/** Mgrid simplifie pour 3D. */
std::vector<Tensor> mgrid3d(int size);

/** Select elements from a 1D tensor by indices. */
Tensor gather(const Tensor& x, const std::vector<int64_t>& indices);

/** Set element at index in a 1D tensor. */
void scatter(Tensor& x, int64_t idx, const Tensor& value);

/** DCT-II simplifie. */
Tensor dct2(const Tensor& x, int num_coeffs);

// ============================================================================
// 11. API bas niveau "zero-copy" (Phase 1)
//     Les fonctions operent directement sur des pointeurs bruts sans passer
//     par la classe Tensor, evitant ainsi toute copie buffer <-> vector.
//     Utilisees par les wrappers Python via ctypes pour les hot paths.
// ============================================================================

/** relu en place : x[i] = max(0, x[i]) pour i in [0, n). */
void relu_inplace(double* x, int64_t n);

/** sigmoid en place. */
void sigmoid_inplace(double* x, int64_t n);

/** tanh en place. */
void tanh_inplace(double* x, int64_t n);

/** a[i] += b[i] pour i in [0, n). */
void add_inplace(double* a, const double* b, int64_t n);

/** a[i] += s * b[i] pour i in [0, n) (axpy). */
void axpy_inplace(double* a, const double* b, int64_t n, double s);

/** a[i] *= s pour i in [0, n). */
void scale_inplace(double* a, int64_t n, double s);

/** matmul 2D dans un buffer pre-alloue : C[M*N] = A[M*K] @ B[K*N].
 *  Ni allocation ni copie supplementaire. */
void matmul_2d_into(const double* A, int64_t M, int64_t K,
                    const double* B, int64_t N,
                    double* C);

/** convolve2d dans un buffer pre-alloue.
 *  img[ih*iw], kernel[kh*kw], out[(ih-kh+1)*(iw-kw+1)]. */
void convolve2d_into(const double* img, int64_t ih, int64_t iw,
                     const double* kernel, int64_t kh, int64_t kw,
                     double* out);

} // namespace ia_core

#endif // IA_ENGINE_H