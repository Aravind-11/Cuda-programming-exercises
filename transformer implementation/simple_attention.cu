
import numpy as np
np.random.seed(0)

# -------------------- tiny config (single-head for clarity) --------------------
B, T, D = 1, 2, 4        # batch, seq-length, model-dim (small for readability)
d_k = D                  # single head uses full D for simplicity

# -------------------- simple helpers --------------------
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))

def gelu_grad(x):
    a = np.sqrt(2/np.pi)
    u = a*(x + 0.044715*x**3)
    tanh_u = np.tanh(u)
    sech2 = 1 - tanh_u**2
    du_dx = a*(1 + 3*0.044715*x**2)
    return 0.5*(1 + tanh_u) + 0.5*x*sech2*du_dx

def layer_norm_forward(x, eps=1e-5):
    # x: (B,T,D) per-token LN across features
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0/np.sqrt(var + eps)
    x_norm = (x - mu)*inv
    return x_norm, (x, mu, var, inv, eps)

def layer_norm_backward(dy, cache):
    # standard LN backward (per-token)
    x, mu, var, inv, eps = cache
    N = x.shape[-1]
    x_mu = x - mu
    dx_norm = dy
    dvar = np.sum(dx_norm * x_mu * -0.5 * inv**3, axis=-1, keepdims=True)
    dmu = np.sum(dx_norm * -inv, axis=-1, keepdims=True) + dvar * np.sum(-2.0 * x_mu, axis=-1, keepdims=True)/N
    dx = dx_norm * inv + dvar * 2.0 * x_mu / N + dmu / N
    return dx

# -------------------- params (random small init) --------------------
W_Q = np.random.randn(D, D) * 0.1
W_K = np.random.randn(D, D) * 0.1
W_V = np.random.randn(D, D) * 0.1
W_O = np.random.randn(D, D) * 0.1

W1 = np.random.randn(D, 2*D) * 0.1   # FFN expand
b1 = np.zeros(2*D)
W2 = np.random.randn(2*D, D) * 0.1
b2 = np.zeros(D)

gamma1 = np.ones(D); beta1 = np.zeros(D)   # LN after input (scale/shift)
gamma2 = np.ones(D); beta2 = np.zeros(D)   # LN after attn/residual

# -------------------- toy input + target --------------------
x = np.random.randn(B, T, D)
target = np.random.randn(B, T, D)

# -------------------- FORWARD PASS --------------------
cache = {}

# Pre-LN (simplified: LN on input)
x_ln, ln1_cache = layer_norm_forward(x)                 # x_norm = (x - mu)/sqrt(var)
x_ln = x_ln * gamma1 + beta1                            # scale & shift
cache['ln1'] = (ln1_cache, x_norm)  # cache both! (CLAUDE CHANGE)


# Q, K, V linear projections
Q = x_ln @ W_Q               # (B,T,D)
K = x_ln @ W_K
V = x_ln @ W_V
cache['Q,K,V'] = (Q, K, V)

# Single-head scaled dot-product attention
scores = Q @ K.transpose(0,2,1) / np.sqrt(d_k)   # (B,T,T)
A = softmax(scores, axis=-1)                      # attention weights
head = A @ V                                      # (B,T,D)
cache['scores,A,head'] = (scores, A, head)

# Output projection
attn_out = head @ W_O            # (B,T,D)
cache['attn_out'] = attn_out

# First residual: x + attn_out
res1 = x + attn_out
cache['res1'] = res1.copy()

# LN before FFN (post-attn LN)
res1_ln, ln2_cache = layer_norm_forward(res1)
res1_ln = res1_ln * gamma2 + beta2
cache['ln2'] = (ln2_cache, res1_norm)

# FFN: expand -> gelu -> project back
h = res1_ln @ W1 + b1       # (B,T,2D)
h_act = gelu(h)
ffn_out = h_act @ W2 + b2   # (B,T,D)
cache['h,h_act,ffn_out'] = (h, h_act, ffn_out)

# final residual -> output
out = res1 + ffn_out         # (B,T,D)

# MSE loss (mean over all elements)
loss = 0.5 * np.mean((out - target)**2)
print("forward loss:", loss)

# -------------------- BACKWARD PASS (manual) --------------------
grads = {name: np.zeros_like(val) for name,val in [
    ("W_Q",W_Q),("W_K",W_K),("W_V",W_V),("W_O",W_O),
    ("W1",W1),("b1",b1),("W2",W2),("b2",b2),
    ("gamma1",gamma1),("beta1",beta1),("gamma2",gamma2),("beta2",beta2)
]}

# dLoss/dout: derivative of 0.5*mean((out-target)^2) -> (out-target)/N_total
N_total = out.size
dout = (out - target) / N_total

# -- final residual out = res1 + ffn_out
dres1 = dout.copy()        # gradient through identity/residual
dffn = dout.copy()         # gradient into FFN output

# -- FFN backward: ffn_out = h_act @ W2 + b2
grads['W2'] += h_act.reshape(-1, h_act.shape[-1]).T @ dffn.reshape(-1, D)   # dW2 = h_act^T * dffn
grads['b2'] += np.sum(dffn, axis=(0,1))                                    # db2 = sum over batch & time

# backprop into h_act
dh_act = dffn @ W2.T                   # (B,T,2D)
dh = dh_act * gelu_grad(h)             # elementwise through GELU

# W1, b1 grads: h = res1_ln @ W1 + b1
grads['W1'] += res1_ln.reshape(-1, D).T @ dh.reshape(-1, 2*D)   # dW1 = res1_ln^T * dh
grads['b1'] += np.sum(dh, axis=(0,1))                         # db1 = sum

# gradient into res1_ln from FFN
dres1_ln_from_ffn = dh @ W1.T      # (B,T,D)

# -- LN2 backward: res1_ln = LN(res1) * gamma2 + beta2
# dgamma2 = sum( dres1_ln_from_ffn * LN(res1) ), dbeta2 = sum(dres1_ln_from_ffn)
# But LN(res1) (normalized values) = (res1 - mu)/sqrt(var) ; we cached ln2_cache for exact backward.
grads['gamma2'] += np.sum(dres1_ln_from_ffn * ((res1 - ln2_cache[1]) * ln2_cache[3]), axis=(0,1))  # dgamma
grads['beta2']  += np.sum(dres1_ln_from_ffn, axis=(0,1))                                           # dbeta

# scale then LN backward
dln2 = dres1_ln_from_ffn * gamma2            # scale by gamma2 (elementwise)
dx_from_ln2 = layer_norm_backward(dln2, ln2_cache)

# total gradient into res1 = from final residual identity + from LN/FFN path
dres1_total = dres1 + dx_from_ln2

# -- residual1: res1 = x + attn_out
# res1 = x + attn_out, so gradient splits:
#   dL/dx = dL/dres1 (identity branch)
#   dL/dattn_out = dL/dres1 (same gradient flows to both inputs)
dx_resid = dres1_total.copy()    # flows to identity branch (input x)
dattn_out = dres1_total.copy()   # flows into attention output

# -- Attention backward
# attn_out = head @ W_O
grads['W_O'] += head.reshape(-1, D).T @ dattn_out.reshape(-1, D)
dhead = dattn_out @ W_O.T        # gradient into head (B,T,D)

# head = A @ V  -> dA and dV
# By chain rule:
#   dL/dA = dL/dhead @ V^T  (where @ is batch matmul)
#   dL/dV = A^T @ dL/dhead
dA = dhead @ V.transpose(0,2,1)          # (B,T,T) = dhead @ V^T  (for single-head shapes)
dV = A.transpose(0,2,1) @ dhead          # (B,T,D) = A^T @ dhead

# softmax backward: A = softmax(scores)
# dScores = A * (dA - sum(dA * A, axis=-1, keepdims=True))
# Softmax Jacobian: dL/dscores[i,j] = A[i,j] * (dL/dA[i,j] - Σ_k dL/dA[i,k] * A[i,k])
# This comes from: dA/dscores = diag(A) - A*A^T per row
tmp = np.sum(dA * A, axis=-1, keepdims=True)
dScores = A * (dA - tmp)

# scores = Q @ K^T / sqrt(d_k)
dQ = dScores @ K / np.sqrt(d_k)          # dQ = dScores @ K / sqrt
dK = dScores.transpose(0,2,1) @ Q / np.sqrt(d_k)  # dK = dScores^T @ Q / sqrt

# Q = x_ln @ W_Q  etc -> compute W grads and dx_ln contribution
grads['W_Q'] += x_ln.reshape(-1, D).T @ dQ.reshape(-1, D)
grads['W_K'] += x_ln.reshape(-1, D).T @ dK.reshape(-1, D)
grads['W_V'] += x_ln.reshape(-1, D).T @ dV.reshape(-1, D)

dx_ln_from_Q = dQ @ W_Q.T
dx_ln_from_K = dK @ W_K.T
dx_ln_from_V = dV @ W_V.T

dx_ln_attn = dx_ln_from_Q + dx_ln_from_K + dx_ln_from_V   # combined gradient into x_ln from attention

# -- LN1 backward: x_ln = LN(x) * gamma1 + beta1
ln1_cache, x_norm1 = cache['ln1']
grads['gamma1'] += np.sum(dx_ln_attn * x_norm1, axis=(0,1))  # cleaner!
grads['beta1']  += np.sum(dx_ln_attn, axis=(0,1))

dln1 = dx_ln_attn * gamma1
dx_from_ln1 = layer_norm_backward(dln1, ln1_cache)

# total gradient to input x = from LN1 path + from residual identity branch
dx_total = dx_from_ln1 + dx_resid

# -------------------- quick prints --------------------
print("\nParameter gradient norms:")
for k in ['W_Q','W_K','W_V','W_O','W1','W2','b1','b2','gamma1','beta1','gamma2','beta2']:
    print(f" {k:6s}: {np.linalg.norm(grads[k]):.6f}")

# -------------------- GRADIENT CHECK --------------------
def check_grad(param_name, param, grad, eps=1e-5):
    """Finite difference check"""
    orig = param.copy()
    errors = []
    for idx in np.ndindex(param.shape):
        param[idx] = orig[idx] + eps
        loss_plus = compute_loss()  # re-run forward
        param[idx] = orig[idx] - eps
        loss_minus = compute_loss()
        param[idx] = orig[idx]

        numerical = (loss_plus - loss_minus) / (2 * eps)
        analytical = grad[idx]
        errors.append(abs(numerical - analytical))

    print(f"{param_name}: max error = {max(errors):.2e}")


