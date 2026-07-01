/**
 * IA/cpp/autograd.cpp — Implementation de l'autograd tape-based (Phase 4).
 *
 * Design revise : chaque backward_fn capture explicitement son propre id
 * (out_id) pour retrouver son gradient, plutot que de supposer out_id = max(parents)+1.
 */

#include "autograd.h"
#include <unordered_map>

namespace ia_autograd {

// ============================================================================
// Tape global (thread_local pour eviter les courses entre threads Python)
// ============================================================================

struct TapeNode {
    std::vector<int64_t> parent_ids;          // ids des Vars inputs
    std::function<void()> backward_fn;        // remplit .grad des parents
};

namespace {
thread_local std::vector<TapeNode> g_tape;
thread_local std::vector<std::shared_ptr<Var>> g_vars;  // propriete des Vars
}

// ============================================================================
// Var
// ============================================================================

Var::Var(ia_core::Tensor t) : data(std::move(t)), id(-1) {
    id = static_cast<int64_t>(g_vars.size());
    auto self = std::make_shared<Var>(*this);
    g_vars.push_back(self);
    TapeNode node;
    node.parent_ids = {};
    node.backward_fn = []() {};
    g_tape.push_back(std::move(node));
}

Var::Var(ia_core::Tensor t, int64_t id_) : data(std::move(t)), id(id_) {}

// ----------------------------------------------------------------------------
// Helpers internes (definis tot pour etre utilisables dans Var::backward)
// ----------------------------------------------------------------------------

std::shared_ptr<Var> get_var_ptr(int64_t id) {
    if (id < 0 || id >= static_cast<int64_t>(g_vars.size())) {
        throw std::runtime_error("autograd: invalid var id");
    }
    return g_vars[id];
}

Var get_var_copy(int64_t id) {
    return *get_var_ptr(id);
}

void Var::backward() {
    // IMPORTANT : this peut etre une copie (cas typique via ag_get_var).
    // On doit operer sur le Var partage dans g_vars, pas sur this.
    auto self = get_var_ptr(id);
    int64_t n = self->data.size();
    self->grad = ia_core::Tensor(ia_core::Shape({n}));
    for (int64_t i = 0; i < n; ++i) self->grad.data[i] = 1.0;

    // Parcourt le tape a l'envers depuis ce noeud.
    for (int64_t i = id; i >= 0; --i) {
        TapeNode& node = g_tape[i];
        if (node.backward_fn) node.backward_fn();
    }
}

// ============================================================================
// Helpers internes (exposes aussi pour le C API via ag_get_var_*)
// ============================================================================

static int64_t push_op(ia_core::Tensor value,
                       std::vector<int64_t> parent_ids,
                       std::function<void(int64_t out_id)> backward_builder) {
    int64_t new_id = static_cast<int64_t>(g_vars.size());
    auto v = std::make_shared<Var>(std::move(value), new_id);
    g_vars.push_back(v);
    TapeNode node;
    node.parent_ids = parent_ids;
    node.backward_fn = [backward_builder, new_id]() {
        backward_builder(new_id);
    };
    g_tape.push_back(std::move(node));
    return new_id;
}

// Helper pour recuperer le grad d'un Var (initialise a 0 si vide)
static ia_core::Tensor& ensure_grad(int64_t id) {
    auto v = get_var_ptr(id);
    if (v->grad.size() == 0) {
        v->grad = ia_core::Tensor(v->data.shape);
    }
    return v->grad;
}

// ============================================================================
// Operations
// ============================================================================

Var relu(const Var& x) {
    ia_core::Tensor val = ia_core::relu(x.data);
    int64_t x_id = x.id;

    auto bb = [x_id](int64_t out_id) {
        auto x_var = get_var_ptr(x_id);
        auto out_var = get_var_ptr(out_id);
        auto& xg = ensure_grad(x_id);
        for (int64_t i = 0; i < x_var->data.size(); ++i) {
            double deriv = x_var->data.data[i] > 0.0 ? 1.0 : 0.0;
            xg.data[i] += deriv * out_var->grad.data[i];
        }
    };

    int64_t new_id = push_op(std::move(val), {x_id}, std::move(bb));
    return *get_var_ptr(new_id);
}

Var sigmoid(const Var& x) {
    ia_core::Tensor s = ia_core::sigmoid(x.data);
    int64_t x_id = x.id;

    auto bb = [x_id, s](int64_t out_id) {
        auto out_var = get_var_ptr(out_id);
        auto& xg = ensure_grad(x_id);
        for (int64_t i = 0; i < s.size(); ++i) {
            double si = s.data[i];
            double deriv = si * (1.0 - si);
            xg.data[i] += deriv * out_var->grad.data[i];
        }
    };

    int64_t new_id = push_op(std::move(s), {x_id}, std::move(bb));
    return *get_var_ptr(new_id);
}

Var tanh(const Var& x) {
    ia_core::Tensor t = ia_core::tanh_act(x.data);
    int64_t x_id = x.id;

    auto bb = [x_id, t](int64_t out_id) {
        auto out_var = get_var_ptr(out_id);
        auto& xg = ensure_grad(x_id);
        for (int64_t i = 0; i < t.size(); ++i) {
            double ti = t.data[i];
            double deriv = 1.0 - ti * ti;
            xg.data[i] += deriv * out_var->grad.data[i];
        }
    };

    int64_t new_id = push_op(std::move(t), {x_id}, std::move(bb));
    return *get_var_ptr(new_id);
}

Var add(const Var& a, const Var& b) {
    if (a.data.size() != b.data.size()) {
        throw std::runtime_error("autograd::add: size mismatch");
    }
    ia_core::Tensor val = ia_core::add(a.data, b.data);
    int64_t a_id = a.id, b_id = b.id;

    auto bb = [a_id, b_id](int64_t out_id) {
        auto out_var = get_var_ptr(out_id);
        auto& ag = ensure_grad(a_id);
        auto& bg = ensure_grad(b_id);
        for (int64_t i = 0; i < out_var->grad.size(); ++i) {
            ag.data[i] += out_var->grad.data[i];
            bg.data[i] += out_var->grad.data[i];
        }
    };

    int64_t new_id = push_op(std::move(val), {a_id, b_id}, std::move(bb));
    return *get_var_ptr(new_id);
}

Var sub(const Var& a, const Var& b) {
    if (a.data.size() != b.data.size()) {
        throw std::runtime_error("autograd::sub: size mismatch");
    }
    ia_core::Tensor val = ia_core::sub(a.data, b.data);
    int64_t a_id = a.id, b_id = b.id;

    auto bb = [a_id, b_id](int64_t out_id) {
        auto out_var = get_var_ptr(out_id);
        auto& ag = ensure_grad(a_id);
        auto& bg = ensure_grad(b_id);
        for (int64_t i = 0; i < out_var->grad.size(); ++i) {
            ag.data[i] += out_var->grad.data[i];
            bg.data[i] -= out_var->grad.data[i];
        }
    };

    int64_t new_id = push_op(std::move(val), {a_id, b_id}, std::move(bb));
    return *get_var_ptr(new_id);
}

Var mul(const Var& a, const Var& b) {
    if (a.data.size() != b.data.size()) {
        throw std::runtime_error("autograd::mul: size mismatch");
    }
    ia_core::Tensor val = ia_core::mul(a.data, b.data);
    int64_t a_id = a.id, b_id = b.id;

    auto bb = [a_id, b_id](int64_t out_id) {
        auto a_var = get_var_ptr(a_id);
        auto b_var = get_var_ptr(b_id);
        auto out_var = get_var_ptr(out_id);
        auto& ag = ensure_grad(a_id);
        auto& bg = ensure_grad(b_id);
        for (int64_t i = 0; i < out_var->grad.size(); ++i) {
            ag.data[i] += b_var->data.data[i] * out_var->grad.data[i];
            bg.data[i] += a_var->data.data[i] * out_var->grad.data[i];
        }
    };

    int64_t new_id = push_op(std::move(val), {a_id, b_id}, std::move(bb));
    return *get_var_ptr(new_id);
}

Var scale(const Var& x, double s) {
    ia_core::Tensor val = ia_core::scale(x.data, s);
    int64_t x_id = x.id;

    auto bb = [x_id, s](int64_t out_id) {
        auto out_var = get_var_ptr(out_id);
        auto& xg = ensure_grad(x_id);
        for (int64_t i = 0; i < out_var->grad.size(); ++i) {
            xg.data[i] += s * out_var->grad.data[i];
        }
    };

    int64_t new_id = push_op(std::move(val), {x_id}, std::move(bb));
    return *get_var_ptr(new_id);
}

Var matmul(const Var& A, const Var& B) {
    if (A.data.ndim() != 2 || B.data.ndim() != 2) {
        throw std::runtime_error("autograd::matmul: only 2D supported");
    }
    int64_t M = A.data.shape[0], K = A.data.shape[1];
    int64_t K2 = B.data.shape[0], N = B.data.shape[1];
    if (K != K2) throw std::runtime_error("autograd::matmul: K mismatch");
    ia_core::Tensor val = ia_core::matmul(A.data, B.data);
    int64_t a_id = A.id, b_id = B.id;

    auto bb = [a_id, b_id, M, K, N](int64_t out_id) {
        auto A_var = get_var_ptr(a_id);
        auto B_var = get_var_ptr(b_id);
        auto out_var = get_var_ptr(out_id);
        auto& Ag = ensure_grad(a_id);
        auto& Bg = ensure_grad(b_id);

        const double* dC = out_var->grad.data.data();
        const double* Aptr = A_var->data.data.data();
        const double* Bptr = B_var->data.data.data();
        double* dA = Ag.data.data();
        double* dB = Bg.data.data();

        // dA = dC @ B^T
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t k = 0; k < K; ++k) {
                double s = 0.0;
                for (int64_t j = 0; j < N; ++j) {
                    s += dC[i * N + j] * Bptr[k * N + j];
                }
                dA[i * K + k] += s;
            }
        }
        // dB = A^T @ dC
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t j = 0; j < N; ++j) {
                double s = 0.0;
                for (int64_t i = 0; i < M; ++i) {
                    s += Aptr[i * K + k] * dC[i * N + j];
                }
                dB[k * N + j] += s;
            }
        }
    };

    int64_t new_id = push_op(std::move(val), {a_id, b_id}, std::move(bb));
    return *get_var_ptr(new_id);
}

Var mse(const Var& pred, const Var& target) {
    if (pred.data.size() != target.data.size()) {
        throw std::runtime_error("autograd::mse: size mismatch");
    }
    double loss = ia_core::mse_loss(pred.data, target.data);
    ia_core::Tensor val({loss}, ia_core::Shape({1}));
    int64_t p_id = pred.id, t_id = target.id;
    int64_t n = pred.data.size();

    auto bb = [p_id, t_id, n](int64_t out_id) {
        auto p_var = get_var_ptr(p_id);
        auto t_var = get_var_ptr(t_id);
        auto out_var = get_var_ptr(out_id);
        auto& pg = ensure_grad(p_id);
        double g = out_var->grad.data[0];
        for (int64_t i = 0; i < n; ++i) {
            pg.data[i] += g * 2.0 * (p_var->data.data[i] - t_var->data.data[i]) / static_cast<double>(n);
        }
        // target est typiquement une feuille sans require_grad
    };

    int64_t new_id = push_op(std::move(val), {p_id, t_id}, std::move(bb));
    return *get_var_ptr(new_id);
}

Var dot(const Var& a, const Var& b) {
    if (a.data.ndim() != 1 || b.data.ndim() != 1 || a.data.size() != b.data.size()) {
        throw std::runtime_error("autograd::dot: requires 1D same-size");
    }
    double v = ia_core::dot1d(a.data, b.data);
    ia_core::Tensor val({v}, ia_core::Shape({1}));
    int64_t a_id = a.id, b_id = b.id;
    int64_t n = a.data.size();

    auto bb = [a_id, b_id, n](int64_t out_id) {
        auto a_var = get_var_ptr(a_id);
        auto b_var = get_var_ptr(b_id);
        auto out_var = get_var_ptr(out_id);
        auto& ag = ensure_grad(a_id);
        auto& bg = ensure_grad(b_id);
        double g = out_var->grad.data[0];
        for (int64_t i = 0; i < n; ++i) {
            ag.data[i] += g * b_var->data.data[i];
            bg.data[i] += g * a_var->data.data[i];
        }
    };

    int64_t new_id = push_op(std::move(val), {a_id, b_id}, std::move(bb));
    return *get_var_ptr(new_id);
}

// ============================================================================
// Gestion du tape
// ============================================================================

void reset_tape() {
    g_tape.clear();
    g_vars.clear();
}

int64_t tape_size() {
    return static_cast<int64_t>(g_tape.size());
}

} // namespace ia_autograd

// ============================================================================
// Helpers exposes pour le C API (extern "C" dans c_api.cpp)
// ============================================================================

ia_autograd::Var ag_get_var(int64_t id) {
    return ia_autograd::get_var_copy(id);
}

const double* ag_get_var_data_ptr(int64_t id, int64_t* out_shape, int* out_ndim) {
    auto v = ia_autograd::get_var_ptr(id);
    *out_ndim = static_cast<int>(v->data.ndim());
    for (int i = 0; i < *out_ndim; ++i) out_shape[i] = v->data.shape[i];
    return v->data.data.data();
}

const double* ag_get_var_grad_ptr(int64_t id, int64_t* out_shape, int* out_ndim) {
    auto v = ia_autograd::get_var_ptr(id);
    if (v->grad.size() == 0) return nullptr;
    *out_ndim = static_cast<int>(v->data.ndim());
    for (int i = 0; i < *out_ndim; ++i) out_shape[i] = v->data.shape[i];
    return v->grad.data.data();
}
