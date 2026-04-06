# Backward Passes in fp8 for Flash Attention 3

## Why this?

There are no public implementations for backward passes in fp8 as far as I know. This is something that would speed up training twofold, saving immense costs.
I also want to actually get good at writing and understanding serious c++ / cuda code, I am just generally interested in how complicated ML systems work and are optimized. 
This is mainly an exercise in my ability to make c++ code that actually works and isn't doing something completely trivial.
My philosophy towards c++ is that the real challenge and skills that are demanded aren't in being able to write out math functions or doing any tricks like that, 
its a test of raw programming skills like memory management and understanding abstractions, skills that will never go out of style and I personally just want to master. 

## What is Flash Attention, and why does it work? + History Lesson

The core idea behind the Flash Attention algorithm is the following:
> **Compute is cheap and memory access is expensive**
###
Attention is one of the core mechanisms of modern ML, so it's important to understand what it is and its implementation. Often it is compactly written as:
$$
\text{Attention} = \text{Softmax}\left( \frac{QK^T}{\sqrt{d_{\text{model}}}}\right)V
$$
Less aesthetically but more mathematically we can write:
$$
A(x) = \text{Softmax}\left( \frac{(x\cdot W_Q ) \cdot (x\cdot W_K)^T}{\sqrt{d_{\text{model}}}} \right) (x \cdot W_V)
$$
Where $x$ is a sequence of $N$ vectors in $\mathbb{R}^d$, and $W_Q,W_K,W_V$ are our weight matrices.
The issue that flash attention solves is the quadratic scale-up; $QK^T$ is an $N\times N$ matrix; it grows quaratically with sequence length. There are two levels of GPU memory that are relevant for this discussion, High Bandwidth Memory (HBM) and Static RAM (SRAM) or Shared memory. HBM is large but slow to access, SRAM is small but easy to access. We never want to materialize $QK^T$ inside of HBM since it would be too slow to operate on.
#### Flash Attention 1
Flash Attention 1 works by starting with our Q,K,V matrices in HBM, and tiling them into smaller matrices. These tiles are loaded into SRAM, and a partial attention is computed for each. The softmax is accumulated and we loop over all the tiles, until we finish.