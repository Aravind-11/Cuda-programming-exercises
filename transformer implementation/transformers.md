# Gaussian Error Linear Unit (GELU)

This document explains the forward and backward passes for both the **exact** and the **tanh-approximate** GELU activations, using GitHub-compatible MathJax formatting.

---

# 1. Overview

The GELU activation is defined as:

$$
\text{GELU}(x) = x \, \Phi(x)
$$

Where:

- $\Phi(x)$ is the **standard normal CDF**
- $\phi(x)$ is the **standard normal PDF**

---

# 2. Exact GELU

## 2.1 Forward Pass

Exact GELU can be written as:

$$
\text{GELU}(x) = x \, \Phi(x)
$$

Using the error function:

$$
\Phi(x) = \frac{1}{2}\left(1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$

Thus:

$$
\text{GELU}(x)
=
\frac{x}{2}\left(1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$

---

## 2.2 Backward Pass (Exact Derivative)

Given:

$$
y = x \Phi(x)
$$

Derivative using product rule:

$$
\frac{dy}{dx}
=
\Phi(x) + x \Phi'(x)
$$

But:

$$
\Phi'(x) = \phi(x)
=
\frac{1}{\sqrt{2\pi}}
e^{-x^2/2}
$$

Final exact gradient:

$$
\boxed{
\text{GELU}'(x) = \Phi(x) + x \phi(x)
}
$$

---

# 3. Approximate GELU (Tanh Approximation)

A computationally efficient version:

$$
\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(
\sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)
\right)\right)
$$

Define the inner term:

$$
t = \sqrt{\frac{2}{\pi}}(x + 0.044715x^3)
$$

---

## 3.1 Backward Pass (Approx Derivative)

Using derivative of $\tanh$:

$$
\frac{d}{dx}\tanh(t) = (1 - \tanh^2(t)) \cdot t'
$$

Where:

$$
t' = \sqrt{\frac{2}{\pi}} \left(1 + 3 \cdot 0.044715 x^2\right)
$$

Full approximate derivative:

$$
\boxed{
\text{GELU}'_{\text{approx}}(x)
=
0.5(1+\tanh(t))
+
0.5x(1 - \tanh^2(t))\, t'
}
$$

---

# 4. Summary Table

| Version | Forward | Backward |
|--------|---------|----------|
| **Exact GELU** | $\text{GELU}(x) = x\Phi(x)$ | $\text{GELU}'(x) = \Phi(x) + x\phi(x)$ |
| **Tanh Approx** | $0.5x(1+\tanh(t))$ | $0.5(1+\tanh(t)) + 0.5x(1-\tanh^2(t))t'$ |

---

# 5. Notes

- The tanh version is widely used in Transformers due to speed.
- Both exact and approximate forms are smooth and differentiable.
- Exact version requires computing the error function $\operatorname{erf}$.

