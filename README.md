# kittention
kittention: the attention archive

![Kittention Banner](./kittention.png)

## Overview

**kittention** is a modular collection for different **attention mechanisms** implemented in numpy.  
The is just a fun side project.

---

## Features

- Unified implementation interface for various attention types  
- Will contain all attention types starting from 2014

---

## Attention Types

Implementation status:

## Attention Types

| Attention Mechanism | Paper | Status |
|--------------------|-------|--------|
| Self Attention (Baseline) | https://arxiv.org/abs/1706.03762 | ✅ |
| Sparse Attention | https://arxiv.org/abs/1904.10509 | ✅ |
| Grouped Query Attention | https://arxiv.org/abs/2305.13245 | ✅ |
| Linear Attention | (tba) | ⏳ |
| Multi-Head Latent Attention | https://arxiv.org/abs/2405.04434 | ⏳ |
| Sparse Sinkhorn Attention | https://arxiv.org/pdf/2002.11296 | ✅ |
| Spark Attention | https://arxiv.org/pdf/2506.06644 | ✅ |
| SeedFold Linear Attention 🧬| https://arxiv.org/abs/2512.24354 | ⏳ |
| Kimi Delta Attention (kimi-linear) 🧬| https://arxiv.org/pdf/2510.26692 | ⏳ |


## Attention Mechanisms

| Attention Mechanism | Paper | Time | Space | Ops @ L=8192 | Status |
|---------------------|-------|------|-------|--------------|--------|
| Self Attention | [Vaswani et al.](https://arxiv.org/abs/1706.03762) | $O(L^2 \cdot d)$ | $O(L^2)$ | 68.7B | ✅ |
| Multi-Head Attention | [Vaswani et al.](https://arxiv.org/abs/1706.03762) | $O(H \cdot L^2 \cdot d_k)$ | $O(L^2)$ | 68.7B | ✅ |
| Sparse Attention | [Child et al.](https://arxiv.org/abs/1904.10509) | $O(L \cdot \sqrt{L} \cdot d)$* | $O(L \cdot \sqrt{L})$ | ~6.0B | ✅ |
| Grouped Query Attention | [Ainslie et al.](https://arxiv.org/abs/2305.13245) | $O(H \cdot L^2 \cdot d_k)$ | $O(L \cdot G \cdot d_k)$ | 68.7B | ✅ |
| Sparse Sinkhorn Attention | [Tay et al.](https://arxiv.org/pdf/2002.11296) | $O(L \cdot B \cdot d)$ | $O(B^2 + N_B^2)$ | 4.3B | ✅ |
| Spark Attention | [Zhou et al.](https://arxiv.org/pdf/2506.06644) | $O(L^2 \cdot r + L \cdot k \cdot d)$ | $O(L \cdot d)$ | 4.3B | ✅ |
| Kimi Delta Attention | [Kimi Team](https://arxiv.org/pdf/2510.26692) | $O(L \cdot d_k^2)$ | $O(d_k^2)$  | 1.1B | ✅ |
| Linear Attention | TBA |  |  |  | ⏳ |
| Multi-Head Latent Attention | TBA |  |  |  | ⏳ |


---
