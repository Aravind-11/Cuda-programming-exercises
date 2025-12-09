# Gaussian Error Linear Unit (GELU)

This document provides formal definitions and full forward/backward derivations for both the exact and approximate GELU activation functions.

---

## 1. Overview

The Gaussian Error Linear Unit is defined as:

\[
\text{GELU}(x) = x \, \Phi(x)
\]

Where:

- \( \Phi(x) \) is the standard normal CDF  
- \( \phi(x) \) is the standard normal PDF  

---

## 2. Exact GELU

### 2.1 Forward Pass

\[
\text{GELU}(x)
=
x \, \Phi(x)
=
\frac{x}{2}\left(1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
\]

---

### 2.2 Backward Pass (Exact Derivative)

Given:

\[
y = x \Phi(x)
\]

Derivative:

\[
\frac{dy}{dx}
= \Phi(x) + x \Phi'(x)
\]

Using:

\[
\Phi'(x) = \phi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}
\]

Final gradient:

\[
\boxed{
\text{GELU}'(x) = \Phi(x) + x \phi(x)
}
\]

---

## 3. Approximate GELU (Tanh Approximation)

\[
\text{GELU}(x)
\approx
0.5 \, x \left(1 + 
\tanh\left(
\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)
\right)
\right)
\]

Define:

\[
t = \sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)
\]

---

### 3.1 Backward Pass (Approximate Derivative)

\[
\text{GELU}'_{\t_
