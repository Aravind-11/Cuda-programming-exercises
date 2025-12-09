# Backpropagation Through Residual Layers

This document provides a clear, technically precise explanation of how backpropagation works in a residual layer, focusing on the math and intuition that matter in practice.

---

## 1. What a Residual Layer Actually Computes

A standard residual block computes:

1. **Main transformation** $F(x, W)$ — typically a stack of Conv/BN/ReLU or Linear/ReLU layers.
2. **Skip (identity) connection** — it passes $x$ directly to the output.
3. **Output**:

$$y = x + F(x, W)$$

This additive structure is the key to why backprop behaves differently (and more stable) here.

---

## 2. Backprop Through a Residual Block

Assume you know the upstream gradient

$$\frac{\partial L}{\partial y}$$

coming from later layers.

We want to compute:

$$\frac{\partial L}{\partial x} \quad \text{and} \quad \frac{\partial L}{\partial W}$$

---

### 2.1 Gradient w.r.t. input $x$

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

---

### 2.2 Gradient w.r.t. the residual block weights $W$

Normal chain rule:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial F(x, W)}{\partial W}$$

Nothing special here: only the residual branch has learnable weights.

---

## 3. Why Residual Blocks Make Deep Nets Trainable

Because one part of the gradient bypasses the nonlinear transformation entirely:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} + \text{(something potentially small)}$$

Even if $\frac{\partial F}{\partial x}$ becomes tiny in deep stacks, the identity term always remains 1.

This prevents gradients from collapsing to zero through very deep models.

---

## 4. Key Takeaways

- **Skip connection provides gradient highway**: The term $\frac{\partial L}{\partial y}$ flows directly back without multiplication by layer weights
- **Residual path adds refinement**: The term $\frac{\partial L}{\partial y} \cdot \frac{\partial F}{\partial x}$ provides additional gradient signal through the learned transformation
- **Stability in depth**: Even if $\frac{\partial F}{\partial x} \approx 0$ in some layers, gradient flow is maintained via the identity path
- **No vanishing gradient problem**: Unlike plain deep networks where gradients multiply through many layers, ResNets guarantee at least one direct path

---


```python
# Numpy example: forward + manual backward pass for a simple residual block
# Model:
#   z1 = W1 @ x + b1
#   a1 = relu(z1)
#   F = W2 @ a1 + b2
#   y = x + F
# Loss:  L = 0.5 * ||y - target||^2
#
# Computes analytic gradients and checks via finite differences.

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

---


# Gaussian Error Linear Unit (GELU)
This explains the forward and backward passes for both the exact and approximate GELU activations, using GitHub-compatible LaTeX.

---

## 1. Overview
The GELU activation is defined as:

$$\text{GELU}(x) = x \, \Phi(x)$$

Where:
- $\Phi(x)$ is the standard normal cumulative distribution function (CDF)  
- $\phi(x)$ is the standard normal probability density function (PDF)  

---

## 2. Exact GELU

### 2.1 Forward Pass
Exact GELU:

$$\text{GELU}(x) = x \, \Phi(x)$$

Where the Gaussian CDF is:

$$\Phi(x) = \frac{1}{2}\left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$

Thus:

$$\text{GELU}(x) = \frac{x}{2}\left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$

---

### 2.2 Backward Pass (Exact Derivative)
Given:

$$y = x \, \Phi(x)$$

Differentiate:

$$\frac{dy}{dx} = \Phi(x) + x \Phi'(x)$$

Using the PDF of a normal distribution:

$$\Phi'(x) = \phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$$

Final exact gradient:

$$\text{GELU}'(x) = \Phi(x) + x \phi(x)$$

---

## 3. Approximate GELU (Tanh Approximation)
A commonly used approximation:

$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)\right)\right)$$

Define:

$$t = \sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)$$

---

### 3.1 Backward Pass (Approx Derivative)
Derivative of tanh:

$$\frac{d}{dx}\tanh(t) = (1 - \tanh^2(t)) \cdot t'$$

Where:

$$t' = \sqrt{\frac{2}{\pi}} \left(1 + 3 \cdot 0.044715 x^2\right)$$

Final approximate GELU derivative:

$$\text{GELU}'_{\text{approx}}(x) = 0.5(1 + \tanh(t)) + 0.5x(1 - \tanh^2(t)) \, t'$$

---

## 4. Summary Table

| Version | Forward | Backward |
|---------|---------|----------|
| Exact GELU | $\text{GELU}(x) = x\Phi(x)$ | $\text{GELU}'(x) = \Phi(x) + x\phi(x)$ |
| Approx GELU | $0.5x(1+\tanh(t))$ | $0.5(1+\tanh(t)) + 0.5x(1-\tanh^2(t))t'$ |

---

## 5. Notes
- The tanh approximation is widely used in Transformers for speed.  
- Both exact and approximate forms are smooth and differentiable.  
- Exact GELU requires computing the $\text{erf}$ function.
