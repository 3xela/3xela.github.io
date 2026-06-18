# Jlux : Flux Implementation in Jax Equinox

<object type="image/svg+xml" data="posts/jlux/flow_stream.svg" class="diagram" aria-label="Rectified flow particle stream"></object>

## hook
I used to think that all text to image models were just denoising, but after reimplementing Flux's inference stack in JAX/Equinox, I realized that it's not so simple. Flux.1 Dev is a 12B parameter rectified flow model, and is one of the strongest open weight text-to-image models available. This post walks through Rectified Flow, the Flux.1 Architecture, and the math behind it. 

## background, fast
Denoising diffusion learns to iteratively denoise an image by predicting the noise added at each step. 
We minimize the expected loss $\mathbb{E}_{\varepsilon \sim \mathcal{N}(0,I)}|| \varepsilon_\theta(x_t, t) - \varepsilon||^2$ with 
$$x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\varepsilon.$$
The premise of RF is that we have an ODE $$\frac{dx}{dt} = v_\theta(x,t)$$ where $v_\theta$ is a vector field that flows from noisy data to images. 
The solution to this ODE gives a **flow** $$\Phi_t(x)$$ which satisfies 
$$v(\Phi_t(x),t) = \frac{d}{dt}\Phi_t(x) $$
That is, to compute $\Phi_t(x)$ we integrate $v(x,t)$.
Given $p_{data}$ and $\mathcal{N}(0,I)$ we learn the ***vector field*** that connects them, $v_\theta(x,t)$ for $t \sim U[0,1]$. 
Let $x_t = (1-t)\cdot x_0 + t\cdot x_1$ for $x_1 \sim \mathcal{N}(0,I)$ and $x_0 \sim p_{data}$. We refer to $(x_0, x_1)$ as a pairing. We minimize the following loss during training: 
$$\mathbb{E}_{t, x_1, x_0} || v_\theta(x_t, t) - (x_1 - x_0) ||^2$$  

Note however, that the interpolants $x_t$ are not guaranteed to be unique whenever
$$(1-t)x_0^A + t x_1^A = (1-t) x_0^B + t x_1^B$$ 
for two different pairings $(x_0^A, x_1^A), (x_0^B, x_1^B)$. Since $v_\theta$ is a function, it must output one velocity for $(x_t,t)$, so the expectation induces curvature in the vector field if there are any collisions. 

The trick to dealing with this is **reflow** during training. We sample $x_1^\ast \sim \mathcal{N}(0,I)$ and flow along $v_\theta$ to produce $x_0^\ast$, and learn the straight line flow through them. This deals with potential collisions in the $(x,t)$ space. 

## the central question: is RF actually different or is this diffusion with the knobs reset?
Sometimes. ( write this part last tbh, consider the case of flux which is denoising, but RF in general is not denosing. eg between two image distributions.)


## the reimplementation

Why Jax/Equinox? 
Jax provides a framework to writing models as pure mathematical operations. As opposed to pytorch, it is purely functional and has no state changes. Tensor operations get compiled and fused into kernels by the XLA compiler, allowing for optimized inference and training. Equinox is a higher level abstraction over JAX that gives access to common modules such as linear maps, similar to `nn.Module` in pytorch. 

One thing about JAX that I grew to appreciate was `jax.vmap`. In pytorch the canonical experience is dealing with batch sizes as part of tensor shapes. Everything has to act on tensors of shape `[B, s_1, s_2...]`. This is just really inconvenient and makes code prone to shape errors. We don't have to suffer through this when using JAX. We can use `jax.vmap` to turn a function into a batched function. Essentially allowing us to write batch-unaware code. This is probably my favourite thing about JAX. 
I understand `vmap` essentially as a lifting of a function over a tensor product. 
We can think of a tensor as living in $\bigotimes \R^{s_i}$ (tensors of shape `(s_1, ... , s_n)`). Given a function $f : \R^n \to \R^m$ we can apply it to  tensor $x \in \R^n$ to get $f(x)$. If $x$ is batched, i.e. $x \in \R^B \otimes \R^n$. 
our activation doesn't act on this space naturally. So we can define ${vmap}(f)$ as 
$$vmap : (f : \R^n \to \R^m) \to (vmap(f) : \R^B \otimes \R^n \to \R^B \otimes \R^m)$$ with 
$$vmap(f) \sum_{i}e_i \otimes x_i \mapsto \sum_i e_i \otimes f(x_i)$$
for a chosen basis $\{e_i\}$ of $\R^B$ (the axis in JAX). A basis is chosen since mathematically this construction doesn't makes sense for nonlinear $f$.

Why from scratch?
The best way to learn something is to do it yourself. I set out to get a deep understanding of the Flux.1 architecure, and theres no better way than to remake it from scratch. It's easy to use an LLM and just parse codebases, but actually writing and debugging the code yourself will always give you a better understanding of what is going on. 

The backbone of Flux.1 is the Flux diffusion transformer. There are two variants, double block and single block. Classically in latent diffusion (SDXL[^sdxl]) img were assigned to Queries and txt was assigned to Keys and Values respectively, allowing for attention to be computed between them. Flux changes this around by computing attention jointly. Image and Text attend to eachother. Key components of flux:

# AdaLN

Adaptive layernorm, Key component. Allows for a global conditioning from CLIP text embeddings. looks like:
$$AdaLN(x) = (1+\text{scale}) * \text{LayerNorm}(x) + \text{shift}$$

The Scale and shift are computed by a modulation MLP. 

# Double Block

A Flux double block acts on image and text tokens, and conditioned on the text embedding provided by CLIP, guidance and time embedding. Each sequence of tokens gets their own QKV, and attention is computed across their concatenation. Attention outputs are split, modulated and normalized with temb conditioning. 

<object type="image/svg+xml" data="posts/jlux/double_block.svg" class="diagram" aria-label="FluxDoubleStreamBlock forward pass"></object>

# Single Block

Single block is more unconventional. input is temb and text + image tokens concatenated. After a linear, split into a self attention channel and an MLP. concat then condition on 

<object type="image/svg+xml" data="posts/jlux/single_block.svg" class="diagram" aria-label="FluxSingleStreamBlock forward pass"></object>

# RoPE

RoPE teaches a model what goes where. Attention is permutation equivariant i.e. $\sigma \in S_n, \sigma(Attn(x)) = Attn(\sigma(x))$. RoPE solves this by rotating pairs of components, preserving relative positions but introducing spatial dependence. Its just an isometry on $\mathbb{R}^n$ i.e. looks like a block diagonal matrix with blocks:
$$\begin{bmatrix} \cos \theta & -\sin \theta \\  \sin \theta & \cos \theta \end{bmatrix}$$. 
the point is that $$\langle R_n q , R_m k  \rangle = \langle q, R_{n-m}k \rangle.$$
In flux this RoPE is lifted to account for the `(0, height_patch, width_patch)` input shapes.
We want the image patches to know where they are relative to eachother, while the `txt` positional encoding is provided by the T5, which already encodes positional information.
Decompose as 
$$\R^{3n} = \R^n_{txt} \oplus \R^n_{img_w} \oplus \R^n_{img_h}.$$
and lift rotation to $$I \oplus RoPE $$

<object type="image/svg+xml" data="posts/jlux/rope4.svg" class="diagram" aria-label="RoPE shown as a hue rotation: an image is chopped into patches, each colour-shifts by its position"></object>


## challenges and unexpected events
I overindexed way too hard in reimplementing Flux, I wrote out a multi headed attention myself. Naive attention implementations are memory IO bottlenecked and therefore very slow, due to having to materialize large matrices. So I just ended up using 
`attn_t = jax.nn.dot_product_attention(Q_t, K_t, V_t, implementation="cudnn")`, 
which calls `fused_attention_stablehlo.py`, which routes to flash attention[^flashattn].
It wasn't that simple though. The `cuDNN` backend is very picky and only takes in `fp16/bf16` and several 8-bit quantizations, but somewhere upstream there were `float32` tensors. I ended up having to go on a crazy chase and ended up finding the culprit to be the huggingface modules I wrapped.
- loading weights is weird, nn.sequential in the torch implementation, needs weird hardcoding
As far as I know, there arent any available weights for `Equinox` Flux.1, so I ended up just converting the weights myself. For ease of conversion, all of my Equinox fields needed to match their corresponding modules in the BFL implementation. This is easy to do in general, just loop over the safetensors and turn them into pytrees then just load them into the `Equinox` tree. One exception to this is the `torch.nn.Sequential` at the end of the single DiT blocks, which doesn't naturally map to pytrees. This was circumvented by just hardcoding the case into the weight loader. Not perfect solution but it's code that only needs to run once, so its fine.
Currently we load the weights for the VAE T5 and CLIP from Hf and shuffle data around to get it into jax arrays. I intentionally chose not to reimplement these in `JAX`, since it would derail me from actually working on the actual Flux code. I'll eventually move it all over to raw JAX/Equinox. This would make the code a lot cleaner, and would remove all of the torch dependencies. 

## back to the question
having seen the code: RF is simple because it deserves to be.
the velocity field parameterization + straight-line schedule
is putting in work.

## loose ends
- Need to reimplement Flux.1 Schnell as well. The architecture is basically identical, its just timestep distilled and doesn't take a guidance parameter.
- t5, vae, clip in JAX or use existing implementations. 

[^sdxl]: Podell et al., *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*, [arXiv:2307.01952](https://arxiv.org/abs/2307.01952)
[^flashattn]: Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

<!-- https://alechelbling.com/blog/rectified-flow/ -->