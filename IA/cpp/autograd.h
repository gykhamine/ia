/**
 * IA/cpp/autograd.h — Autograd minimaliste (Phase 4).
 *
 * Design : "Wengert tape" (style tinygrad / micrograd).
 *   - Chaque Var enregistre sur un tape global son operation + ses parents.
 *   - Le backward parcourt le tape a l'envers et accumule les gradients.
 *   - Le user appelle Var::backward() sur la sortie, puis lit .grad sur les
 *     feuilles.
 *
 * Portee (prototype) : relu, sigmoid, add, sub, mul, scale, matmul, mse, dot.
 * Les ops plus complexes (conv, softmax, layer_norm) restent a ajouter dans
 * une deuxieme iteration ; l'API est extensible sans casser le code existant.
 *
 * Important : le tape est global et thread_local ; pas d'utilisation
 * multi-thread concurrente sur un meme tape. C'est volontaire pour rester
 * leger et compatible avec un usage Python via ctypes (GIL protege deja).
 */

#ifndef IA_AUTOGRAD_H
#define IA_AUTOGRAD_H

#include "engine.h"
#include <functional>
#include <vector>
#include <memory>

namespace ia_autograd {

// Forward declaration
struct Var;
struct Tape;

/** Un noeud du graphe de calcul differentiable.
 *  - data : valeur courante (Tensor)
 *  - grad : gradient accumule (Tensor), rempli pendant backward()
 *  - id   : index dans le tape (pour retrouver parents)
 */
struct Var {
    ia_core::Tensor data;
    ia_core::Tensor grad;
    int64_t id;

    Var() : id(-1) {}
    explicit Var(ia_core::Tensor t);
    Var(ia_core::Tensor t, int64_t id_);

    int64_t size() const { return data.size(); }
    const ia_core::Shape& shape_() const { return data.shape; }

    /** Backward depuis ce Var : calcule les grads de tous les parents
     *  enregistres avant lui sur le tape. */
    void backward();
};

/** Operations differentiables. Chacune :
 *  1. calcule la valeur forward (via ia_core::*)
 *  2. push un noeud sur le tape avec un backward_fn
 *  3. retourne le Var resultant
 */
Var relu(const Var& x);
Var sigmoid(const Var& x);
Var tanh(const Var& x);
Var add(const Var& a, const Var& b);
Var sub(const Var& a, const Var& b);
Var mul(const Var& a, const Var& b);          // element-wise
Var scale(const Var& x, double s);
Var matmul(const Var& A, const Var& B);       // 2D x 2D
Var mse(const Var& pred, const Var& target);  // -> Var scalaire (shape {1})
Var dot(const Var& a, const Var& b);          // produit scalaire -> Var

/** Remet le tape a zero. A appeler entre deux forwards. */
void reset_tape();

/** Nombre de noeuds actuellement sur le tape (debug). */
int64_t tape_size();

// ---- Helpers exposes pour le C API ----

/** Retourne une copie d'un Var par id. */
Var get_var_copy(int64_t id);

} // namespace ia_autograd

// Helpers C++ non-namespaces pour le C API (definis dans autograd.cpp)
ia_autograd::Var ag_get_var(int64_t id);
const double* ag_get_var_data_ptr(int64_t id, int64_t* out_shape, int* out_ndim);
const double* ag_get_var_grad_ptr(int64_t id, int64_t* out_shape, int* out_ndim);

#endif // IA_AUTOGRAD_H
