## Backpropagation in Transformer Blocks: What Actually Happens

I spent months implementing Transformers before I really understood what was happening during backpropagation. I could write the forward pass, call `.backward()`, and watch the loss go down—but I didn't truly *get* how gradients flowed through residual connections, or why GELU behaved differently from ReLU, or what made attention's backward pass so intricate.

It wasn't until I sat down and derived everything from scratch—pen, paper, and way too much coffee—that things clicked. And I realized: most people using Transformers every day have never seen this math. They trust PyTorch to handle it, which is fine for building models, but it leaves a gap. You can't debug what you don't understand. You can't design better architectures if you don't know why the current ones work.

So I'm writing this for anyone who's ever wondered what's really going on under the hood. We're going to break down a Transformer block into its core components—residual connections, GELU activations, and scaled dot-product attention—and derive the backward pass for each one. 

If you're the kind of person who feels uneasy using tools you don't fully understand, this is for you.


## What we are computing

A standard Transformer encoder block looks like this:
```
Input x
  ↓
Attention
  ↓
Add & Norm (Residual + LayerNorm)
  ↓
Feed-Forward Network (Linear → GELU → Linear)
  ↓
Add & Norm (Residual + LayerNorm)
  ↓
Output
```

Lets go through each layer one by one. 

---

## Backpropagation Through Residual Layers

### 1. Forward pass

A standard residual block computes:

1. **Main transformation** $F(x, W)$ — typically a stack of Conv/BN/ReLU or Linear/ReLU layers.
2. **Skip (identity) connection** — it passes $x$ directly to the output.
3. **Output**:

$$y = x + F(x, W)$$

This additive structure is the key to why backprop behaves differently (and more stable) here.


### 2. Backprop Through a Residual Block

Assume you know the upstream gradient

$$\frac{\partial L}{\partial y}$$

coming from later layers.

We want to compute:

$$\frac{\partial L}{\partial x} \quad \text{and} \quad \frac{\partial L}{\partial W}$$


#### 2.1 Gradient w.r.t. input $x$

Since

$$y = x + F(x, W),$$

the derivative of $y$ w.r.t. $x$ is:

$$\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$$

Therefore:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(I + \frac{\partial F}{\partial x}\right)$$

This expands to:

$$\frac{\partial L}{\partial x} = \underbrace{\frac{\partial L}{\partial y}}_{\text{skip connection}} + \underbrace{\frac{\partial L}{\partial y} \cdot \frac{\partial F}{\partial x}}_{\text{normal backprop through the main path}}$$

**Interpretation:** The gradient splits into two paths:

1. **Identity path**: gradient flows unchanged

$$\frac{\partial L}{\partial y} \rightarrow \text{back to } x$$

2. **Residual path**: gradient flows through $F$, which may shrink or distort it.

This is the reason ResNets avoid vanishing gradients: the identity connection ensures a clean gradient flow.


#### 2.2 Gradient w.r.t. the residual block weights $W$

Normal chain rule:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial F(x, W)}{\partial W}$$

Nothing special here: only the residual branch has learnable weights.


### 3. Why Residual Blocks Make Deep Nets Trainable

Because one part of the gradient bypasses the nonlinear transformation entirely:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} + \text{(something potentially small)}$$

Even if $\frac{\partial F}{\partial x}$ becomes tiny in deep stacks, the identity term always remains 1.

This prevents gradients from collapsing to zero through very deep models.



```python
# Numpy example: forward + manual backward pass for a simple residual block
# Model:
#   z1 = W1 @ x + b1
#   a1 = relu(z1)
#   F = W2 @ a1 + b2
#   y = x + F
# Loss:  L = 0.5 * ||y - target||^2
#
# Computes gradients and checks via finite differences.

import numpy as np

np.random.seed(0)

# dimensions
in_dim = 3
hid = 4

# random small inputs / params
x = np.random.randn(in_dim)            # input (vector)
target = np.random.randn(in_dim)       # target (vector)
W1 = np.random.randn(hid, in_dim) * 0.1
b1 = np.random.randn(hid) * 0.1
W2 = np.random.randn(in_dim, hid) * 0.1
b2 = np.random.randn(in_dim) * 0.1

def relu(z):
    return np.maximum(0, z)

def forward(x, W1, b1, W2, b2, target):
    z1 = W1.dot(x) + b1         # (hid,)
    a1 = relu(z1)               # (hid,)
    F = W2.dot(a1) + b2         # (in_dim,)
    y = x + F                   # (in_dim,)
    loss = 0.5 * np.sum((y - target)**2)
    cache = (x, z1, a1, F, y)
    return loss, cache

def manual_backward(cache, W1, W2, target):
    x, z1, a1, F, y = cache
    # dL/dy
    dy = y - target             # (in_dim,)
    # y = x + F -> dF = dy, dx from skip = dy
    dF = dy.copy()              # gradient flowing through residual branch
    dx_skip = dy.copy()         # gradient from identity skip connection

    # gradients for W2, b2 from F = W2 @ a1 + b2
    dW2 = np.outer(dF, a1)     # (in_dim, hid)
    db2 = dF.copy()            # (in_dim,)

    # backprop into a1
    da1 = W2.T.dot(dF)         # (hid,)
    dz1 = da1 * (z1 > 0)       # relu backward

    # gradients for W1, b1 from z1 = W1 @ x + b1
    dW1 = np.outer(dz1, x)     # (hid, in_dim)
    db1 = dz1.copy()           # (hid,)

    # gradient into x from residual path
    dx_res = W1.T.dot(dz1)     # (in_dim,)

    # total dx = skip + residual path
    dx = dx_skip + dx_res      # (in_dim,)

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'x': dx, 'y': dy}

# Forward pass
loss, cache = forward(x, W1, b1, W2, b2, target)
grads_manual = manual_backward(cache, W1, W2, target)

print("Loss:", loss)
print("\nManual gradients:")
print("dW1:\n", grads_manual['W1'])
print("db1:\n", grads_manual['b1'])
print("dW2:\n", grads_manual['W2'])
print("db2:\n", grads_manual['b2'])
print("dx:\n", grads_manual['x'])


```



## Gaussian Error Linear Unit (GELU)


The GELU activation is defined as:

$$\text{GELU}(x) = x \, \Phi(x)$$

Where:
- $\Phi(x)$ is the standard normal cumulative distribution function (CDF)  
- $\phi(x)$ is the standard normal probability density function (PDF)  


#### 2.1 Forward Pass
Exact GELU:

$$\text{GELU}(x) = x \, \Phi(x)$$

Where the Gaussian CDF is:

$$\Phi(x) = \frac{1}{2}\left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$

Thus:

$$\text{GELU}(x) = \frac{x}{2}\left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$


#### 2.2 Backward Pass (Exact Derivative)
Given:

$$y = x \, \Phi(x)$$

Differentiate:

$$\frac{dy}{dx} = \Phi(x) + x \Phi'(x)$$

Using the PDF of a normal distribution:

$$\Phi'(x) = \phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$$

Final exact gradient:

$$\text{GELU}'(x) = \Phi(x) + x \phi(x)$$


### 3. Approximate GELU (Tanh Approximation)
A commonly used approximation:

$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)\right)\right)$$

Define:

$$t = \sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)$$


#### 3.1 Backward Pass (Approx Derivative)
Derivative of tanh:

$$\frac{d}{dx}\tanh(t) = (1 - \tanh^2(t)) \cdot t'$$

Where:

$$t' = \sqrt{\frac{2}{\pi}} \left(1 + 3 \cdot 0.044715 x^2\right)$$

Final approximate GELU derivative:

$$\text{GELU}'_{\text{approx}}(x) = 0.5(1 + \tanh(t)) + 0.5x(1 - \tanh^2(t)) \, t'$$


```python

import numpy as np
from math import erf

np.set_printoptions(precision=6, suppress=True)

# -----------------------------
# Standard Normal PDF & CDF
# -----------------------------
def phi(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def Phi(x):
    # Vectorize math.erf for array input
    return 0.5 * (1 + np.vectorize(erf)(x / np.sqrt(2)))

# -----------------------------
# EXACT GELU
# -----------------------------
def gelu_exact(x):
    return x * Phi(x)

def gelu_exact_derivative(x):
    # GELU'(x) = Φ(x) + x * φ(x)
    return Phi(x) + x * phi(x)

# -----------------------------
# TANH APPROX GELU
# -----------------------------
def gelu_tanh(x):
    t = np.sqrt(2/np.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1 + np.tanh(t))

def gelu_tanh_derivative(x):
    t = np.sqrt(2/np.pi) * (x + 0.044715 * x**3)
    tanh_t = np.tanh(t)
    dt_dx = np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * x**2)
    return 0.5 * (1 + tanh_t) + 0.5 * x * (1 - tanh_t**2) * dt_dx



# -----------------------------
# Test inputs
# -----------------------------
x = np.array([-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0])

# Forward
y_exact = gelu_exact(x)
y_tanh  = gelu_tanh(x)

# Backward 
dy_exact = gelu_exact_derivative(x)
dy_tanh  = gelu_tanh_derivative(x)



```


### 5. Notes
- The tanh approximation is widely used in Transformers for speed.  
- Both exact and approximate forms are smooth and differentiable.  
- Exact GELU requires computing the $\text{erf}$ function.



## Scaled Dot-Product Attention: Forward and Backward Pass

This document explains the mathematical derivation and implementation of scaled dot-product attention, including forward pass, backward pass (backpropagation), and numerical gradient verification.


### 1. Overview

Scaled dot-product attention is the core mechanism in Transformer models. Given input sequence $X$, it computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ (Query): $X W_q$, shape $(T, d_k)$
- $K$ (Key): $X W_k$, shape $(T, d_k)$
- $V$ (Value): $X W_v$, shape $(T, d_k)$
- $T$: sequence length
- $d_k$: dimension of queries/keys/values


### 2. Forward Pass

#### 2.1 Step-by-Step Computation

Given input $X \in \mathbb{R}^{T \times d_{\text{model}}}$ and projection matrices $W_q, W_k, W_v \in \mathbb{R}^{d_{\text{model}} \times d_k}$:

1. **Project inputs**:

$$Q = XW_q, \quad K = XW_k, \quad V = XW_v$$

2. **Compute attention logits** (scaled dot-product):

$$Z = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}$$

3. **Apply softmax** (row-wise):

$$A = \text{softmax}(Z) \in \mathbb{R}^{T \times T}$$

where each row $A_i$ sums to 1.

4. **Compute output**:

$$O = AV \in \mathbb{R}^{T \times d_k}$$

#### 2.2 Loss Function

For supervised learning with target $Y \in \mathbb{R}^{T \times d_k}$:

$$L = \frac{1}{2}\sum_{i,j}(O_{ij} - Y_{ij})^2$$


### 3. Backward Pass (Backpropagation)

Working backward through the computational graph:

#### 3.1 Gradient w.r.t. Output

$$\frac{\partial L}{\partial O} = O - Y$$

#### 3.2 Gradient through $O = AV$

Using the product rule:

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial O} V^T \in \mathbb{R}^{T \times T}$$

$$\frac{\partial L}{\partial V} = A^T \frac{\partial L}{\partial O} \in \mathbb{R}^{T \times d_k}$$

#### 3.3 Gradient through Softmax

For row-wise softmax, the Jacobian of row $i$ is:

$$J_i = \text{diag}(A_i) - A_i A_i^T$$

Therefore:

$$\frac{\partial L}{\partial Z_i} = A_i \odot \left(\frac{\partial L}{\partial A_i} - A_i \cdot \frac{\partial L}{\partial A_i}\right)$$

where $\odot$ is element-wise multiplication and $\cdot$ is dot product.

In vectorized form:

$$\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} - A \odot \left(\sum_j \frac{\partial L}{\partial A} \odot A\right)$$

#### 3.4 Gradient through Scaled Dot-Product $Z = \frac{QK^T}{\sqrt{d_k}}$

$$\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial Z} K \in \mathbb{R}^{T \times d_k}$$

$$\frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \left(\frac{\partial L}{\partial Z}\right)^T Q \in \mathbb{R}^{T \times d_k}$$

#### 3.5 Gradient w.r.t. Projection Weights

Since $Q = XW_q$, $K = XW_k$, $V = XW_v$:

$$\frac{\partial L}{\partial W_q} = X^T \frac{\partial L}{\partial Q} \in \mathbb{R}^{d_{\text{model}} \times d_k}$$

$$\frac{\partial L}{\partial W_k} = X^T \frac{\partial L}{\partial K} \in \mathbb{R}^{d_{\text{model}} \times d_k}$$

$$\frac{\partial L}{\partial W_v} = X^T \frac{\partial L}{\partial V} \in \mathbb{R}^{d_{\text{model}} \times d_k}$$

#### 3.6 Gradient w.r.t. Input $X$

The gradient flows through three paths (Q, K, V):

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Q}W_q^T + \frac{\partial L}{\partial K}W_k^T + \frac{\partial L}{\partial V}W_v^T$$


```python
#!/usr/bin/env python3
"""
Scaled Dot-Product Attention (NumPy)
Single-head, single-batch, sequence length T, model dim d_model, projection dim d_k.
Forward + manual backward + numeric gradient checks.
"""
import numpy as np

np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(2)

# -----------------------
# Config / random setup
# -----------------------
T = 4            # sequence length (tokens)
d_model = 6      # input/model dimension
d_k = 3          # key/query/value dim (per head)

# Random small inputs and target
X = rng.normal(scale=0.5, size=(T, d_model))       # (T, d_model)
Wq = rng.normal(scale=0.1, size=(d_model, d_k))    # (d_model, d_k)
Wk = rng.normal(scale=0.1, size=(d_model, d_k))
Wv = rng.normal(scale=0.1, size=(d_model, d_k))
target = rng.normal(scale=0.5, size=(T, d_k))      # supervised target for O

sqrt_dk = np.sqrt(d_k)

# -----------------------
# Utilities
# -----------------------
def softmax_rows(z):
    """Row-wise stable softmax for 2D array z with shape (T, T)."""
    z_max = np.max(z, axis=1, keepdims=True)
    e = np.exp(z - z_max)
    return e / np.sum(e, axis=1, keepdims=True)

# -----------------------
# Forward Pass
# -----------------------
def forward(X, Wq, Wk, Wv, target):
    """
    Computes attention output and loss.
    
    Args:
        X: Input sequence (T, d_model)
        Wq, Wk, Wv: Projection matrices (d_model, d_k)
        target: Supervised target (T, d_k)
    
    Returns:
        loss: Scalar MSE loss
        cache: Tuple of intermediate values for backprop
    """
    Q = X @ Wq               # (T, d_k)
    K = X @ Wk               # (T, d_k)
    V = X @ Wv               # (T, d_k)
    Z = (Q @ K.T) / sqrt_dk  # (T, T)  scaled dot-product logits
    A = softmax_rows(Z)      # (T, T)  attention weights (rows sum to 1)
    O = A @ V                # (T, d_k) output
    
    loss = 0.5 * np.sum((O - target)**2)
    cache = (X, Q, K, V, Z, A, O)
    return loss, cache

# -----------------------
# Manual Backward Pass
# -----------------------
def backward(cache, Wq, Wk, Wv, target):
    """
    Manual backpropagation for scaled dot-product attention.
    
    Key insight for softmax backprop:
    For row-wise softmax, the Jacobian for row i is:
        J_i = diag(A_i) - A_i ⊗ A_i^T
    Therefore:
        dZ_i = A_i ⊙ (dA_i - (A_i · dA_i))
    where · is dot product, ⊙ is element-wise product
    
    Returns:
        dict: Gradients for all parameters and intermediates
    """
    X, Q, K, V, Z, A, O = cache
    
    # Gradient w.r.t. output
    dO = O - target                      # (T, d_k)

    # Backprop through O = A @ V
    dA = dO @ V.T                        # (T, T)
    dV = A.T @ dO                        # (T, d_k)

    # Backprop through softmax (row-wise)
    # For each row i: dZ_i = A_i ⊙ (dA_i - (A_i · dA_i))
    AdotdA = np.sum(A * dA, axis=1, keepdims=True)   # (T, 1)
    dZ = dA - A * AdotdA                             # (T, T)

    # Backprop through Z = (Q @ K^T) / sqrt(d_k)
    dQ = (dZ @ K) / sqrt_dk            # (T, d_k)
    dK = (dZ.T @ Q) / sqrt_dk          # (T, d_k)

    # Gradient w.r.t. projection weights
    dWq = X.T @ dQ                     # (d_model, d_k)
    dWk = X.T @ dK                     # (d_model, d_k)
    dWv = X.T @ dV                     # (d_model, d_k)

    # Gradient w.r.t. input X (sum of three paths: Q, K, V)
    dX = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T   # (T, d_model)

    return {
        'dWq': dWq, 'dWk': dWk, 'dWv': dWv,
        'dX': dX, 'dZ': dZ, 'dA': dA, 'dO': dO,
        'dQ': dQ, 'dK': dK, 'dV': dV
    }

# -----------------------
# Numeric Gradient Check
# -----------------------
def numeric_grad_param(param_name, eps=1e-6):
    """
    Finite-difference gradient check for any parameter.
    
    Args:
        param_name: One of 'Wq', 'Wk', 'Wv', 'X'
        eps: Perturbation size for finite difference
    
    Returns:
        Numerical gradient with same shape as parameter
    """
    if param_name == 'Wq':
        param = Wq
    elif param_name == 'Wk':
        param = Wk
    elif param_name == 'Wv':
        param = Wv
    elif param_name == 'X':
        param = X
    else:
        raise ValueError("unknown param")

    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        orig = param[idx]

        if param_name == 'Wq':
            Wq_p = Wq.copy(); Wq_p[idx] = orig + eps
            Wq_m = Wq.copy(); Wq_m[idx] = orig - eps
            l_plus, _ = forward(X, Wq_p, Wk, Wv, target)
            l_minus, _ = forward(X, Wq_m, Wk, Wv, target)
        elif param_name == 'Wk':
            Wk_p = Wk.copy(); Wk_p[idx] = orig + eps
            Wk_m = Wk.copy(); Wk_m[idx] = orig - eps
            l_plus, _ = forward(X, Wq, Wk_p, Wv, target)
            l_minus, _ = forward(X, Wq, Wk_m, Wv, target)
        elif param_name == 'Wv':
            Wv_p = Wv.copy(); Wv_p[idx] = orig + eps
            Wv_m = Wv.copy(); Wv_m[idx] = orig - eps
            l_plus, _ = forward(X, Wq, Wk, Wv_p, target)
            l_minus, _ = forward(X, Wq, Wk, Wv_m, target)
        elif param_name == 'X':
            X_p = X.copy(); X_p[idx] = orig + eps
            X_m = X.copy(); X_m[idx] = orig - eps
            l_plus, _ = forward(X_p, Wq, Wk, Wv, target)
            l_minus, _ = forward(X_m, Wq, Wk, Wv, target)

        grad[idx] = (l_plus - l_minus) / (2 * eps)
        it.iternext()
    
    return grad

# -----------------------
# Run Tests
# -----------------------
if __name__ == "__main__":
    # Forward pass
    loss, cache = forward(X, Wq, Wk, Wv, target)
    
    # Backward pass
    grads = backward(cache, Wq, Wk, Wv, target)

    print("Loss:", loss)
    print("Shapes: X", X.shape, "Wq", Wq.shape, "Wk", Wk.shape, "Wv", Wv.shape)
    print("Output O shape:", cache[6].shape)
    
    print("\nManual gradient norms:")
    print("||dWq|| {:.6e}  ||dWk|| {:.6e}  ||dWv|| {:.6e}".format(
        np.linalg.norm(grads['dWq']), 
        np.linalg.norm(grads['dWk']), 
        np.linalg.norm(grads['dWv'])))
    print("||dX|| {:.6e}".format(np.linalg.norm(grads['dX'])))

    # Numeric gradient checks
    num_dWq = numeric_grad_param('Wq')
    num_dWk = numeric_grad_param('Wk')
    num_dWv = numeric_grad_param('Wv')
    num_dX  = numeric_grad_param('X')

    print("\nNumeric gradient check (max abs diff):")
    print("dWq diff:", np.max(np.abs(num_dWq - grads['dWq'])))
    print("dWk diff:", np.max(np.abs(num_dWk - grads['dWk'])))
    print("dWv diff:", np.max(np.abs(num_dWv - grads['dWv'])))
    print("dX  diff:", np.max(np.abs(num_dX  - grads['dX'])))

    print("\nSample manual dWq[:2,:2]:\n", grads['dWq'][:2,:2])
    print("Sample numeric dWq[:2,:2]:\n", num_dWq[:2,:2])
```


### Continuation 

Find the attached jupyter notebook in the previous page where I have implemented a full transformer block in numpy and in cuda. The cuda code is implemented using basic functions in c++ and cuda , hence the code is painfully long. Please have some patience to go through the entire code to be able to understand and implement it on your own. Happy Coding!
