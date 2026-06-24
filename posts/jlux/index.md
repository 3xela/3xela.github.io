# [Jlux](https://github.com/3xela/jlux) : Flux Implementation in Jax Equinox

<object type="image/svg+xml" data="posts/jlux/flow_stream.svg" class="diagram" aria-label="Rectified flow particle stream"></object>

## Flux.1
Flux.1 Dev[^flux] is a 12B parameter rectified flow model, and is one of the strongest open weight text-to-image models available. This post walks through Rectified Flow, the Flux.1 Architecture, and the math behind it. 

## Background
Denoising diffusion learns to iteratively denoise an image by predicting the noise added at each step. 
We minimize the expected loss $\mathbb{E}_{\varepsilon \sim \mathcal{N}(0,I)}|| \varepsilon_\theta(x_t, t) - \varepsilon||^2$ with 
$$x_t = \alpha_t x_0 + \sigma_t \varepsilon$$
where $\alpha_t^2 + \sigma_t^2 = 1$. 
The premise of Rectified Flow[^rectflow] (RF) is that we have an ODE $$\frac{dx}{dt} = v_\theta(x,t)$$ where $v_\theta$ is a vector field that flows from noisy data to images. 
The solution to this ODE gives a **flow** $\Phi_t(x)$ which satisfies 
$$v(\Phi_t(x),t) = \frac{d}{dt}\Phi_t(x) $$
That is, to compute $\Phi_t(x)$ we integrate $v(x,t)$.
Given $p_{data}$ and $\mathcal{N}(0,I)$ we learn the ***vector field*** that connects them, $v_\theta(x,t)$ for $t \sim U[0,1]$. 
Let $x_t = (1-t)\cdot x_0 + t\cdot x_1$ for $x_1 \sim \mathcal{N}(0,I)$ and $x_0 \sim p_{data}$. We refer to $(x_0, x_1)$ as a pairing. We minimize the following loss during training: 
$$\mathbb{E}_{t, x_1, x_0} || v_\theta(x_t, t) - (x_1 - x_0) ||^2$$  

Note however, that the interpolants $x_t$ are not guaranteed to be unique whenever
$$(1-t)x_0^A + t x_1^A = (1-t) x_0^B + t x_1^B$$ 
for two different pairings $(x_0^A, x_1^A), (x_0^B, x_1^B)$. Since $v_\theta$ is a function, it must output one velocity for $(x_t,t)$, so the expectation induces curvature in the vector field if there are any collisions. 

The trick to dealing with this is **reflow** during training. We sample $x_1^\ast \sim \mathcal{N}(0,I)$ and flow along $v_\theta$ to produce $x_0^\ast$, and learn the straight line flow through them. This deals with potential collisions in the $(x,t)$ space. 

## Is RF just denoising?
Under some assumptions, it can be. Here [^gaoflow] they show the equivalence of these two frameworks. For a more in depth treatment of RF I highly reccomend reading [this blog](https://alechelbling.com/blog/rectified-flow/). The idea is that for a denoising diffusion process, we can write
$$\tilde{x}_s = \tilde{x}_t + f\cdot(\eta_s - \eta_t) $$
Where $f$ is our neural network. Different choices of $\tilde{x}_t$ and $\eta_t$ recover either case. In the case of denoising, $\tilde{x}_t = \frac{x_t}{\alpha_t}$, $\eta_t = \frac{\sigma_t}{\alpha_t}$, and RF $\tilde{x}_t = x_t$, $\eta_t = t$. In the case of Flux however, and other RF models, $t$ is not sampled from a uniform distribution. It doesn't make sense to train a model to recover an image for very small $t$ or $t$ close to 1. 
There just isnt really anything to be learned. Instead we use a parametrization of $t$ with a larger mid-range mass. The Flux schedule is defined as
$$t\mapsto \frac{t e^{\mu(N)}}{1 + t \left( e^{\mu(N)}-1 \right)}$$
Where $\mu(N)$ is a function on the number of image tokens.
Earlier I implied that Rectified Flow isnt always denoising, just under some assumption it can be shown to be a reparametrized denoising process. In general, RF is valid for any two arbitrary distributions e.g. the distribution of dog and cat images.

## The Reimplementation

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
The best way to learn something is to do it yourself. I set out to get a deep understanding of the Flux.1 architecure, and theres no better way than to remake it from scratch. It's easy to use an LLM and just vibe code a repo, but actually writing and debugging the code yourself will always give you a better understanding of what is going on. 

## Key Components

The backbone of Flux.1 is the Flux diffusion transformer. There are two variants, double block and single block. Classically in latent diffusion (SDXL[^sdxl]) img were assigned to Queries and txt was assigned to Keys and Values respectively, allowing for attention to be computed between them. Flux changes this around by computing attention jointly. Image and Text attend to eachother. Key components of flux:

### AdaLN

Adaptive layernorm, Key component. Allows for a global conditioning from CLIP text embeddings. looks like:
$$AdaLN(x) = (1+\text{scale}) * \text{LayerNorm}(x) + \text{shift}$$

The Scale and shift are computed by a modulation MLP. 

### Double Block:

A Flux double block acts on image and text tokens, and conditioned on the text embedding provided by CLIP, guidance and time embedding. Each sequence of tokens gets their own QKV, and attention is computed across their concatenation. Attention outputs are split, modulated and normalized with temb conditioning. 

<object type="image/svg+xml" data="posts/jlux/double_block.svg" class="diagram" aria-label="FluxDoubleStreamBlock forward pass"></object>

### Single Block

Single block is more unconventional. input is temb and text + image tokens concatenated. After a linear, split into a self attention channel and an MLP.

<object type="image/svg+xml" data="posts/jlux/single_block.svg" class="diagram" aria-label="FluxSingleStreamBlock forward pass"></object>

### RoPE

RoPE teaches a model what goes where. Attention is permutation equivariant i.e. $\sigma \in S_n, \sigma(Attn(x)) = Attn(\sigma(x))$. RoPE solves this by rotating pairs of components, preserving relative positions but introducing spatial dependence. Its just an isometry on $\mathbb{R}^n$ i.e. looks like a block diagonal matrix with blocks:
$$\begin{bmatrix} \cos \theta & -\sin \theta \\  \sin \theta & \cos \theta \end{bmatrix}.$$
the point is that $$\langle R_n q , R_m k  \rangle = \langle q, R_{n-m}k \rangle.$$
In flux this RoPE is lifted to account for the `(0, height_patch, width_patch)` input shapes.
We want the image patches to know where they are relative to eachother, while the `txt` positional encoding is provided by the T5, which already encodes positional information.
Decompose as 
$$\R^{128} = \R^{16}_{txt} \oplus \R^{56}_{img_w} \oplus \R^{56}_{img_h}.$$
and lift rotation to $$I \oplus RoPE $$

<object type="image/svg+xml" data="posts/jlux/rope4.svg" class="diagram" aria-label="RoPE shown as a hue rotation: an image is chopped into patches, each colour-shifts by its position"></object>


## Challenges and Unexpected Events
I overindexed way too hard in reimplementing Flux, I wrote out a multi headed attention myself. Naive attention implementations are memory IO bottlenecked and therefore very slow, due to having to materialize large matrices. So I just ended up using 
`attn_t = jax.nn.dot_product_attention(Q_t, K_t, V_t, implementation="cudnn")`, 
which calls `fused_attention_stablehlo.py`, which routes to flash attention[^flashattn].
It wasn't that simple though. The `cuDNN` backend is very picky and only takes in `fp16/bf16` and several 8-bit quantizations, but somewhere upstream there were `float32` tensors. I ended up having to go on a crazy chase and ended up finding the culprit to be the huggingface modules I wrapped.


As far as I know, there arent any available weights for `Equinox` Flux.1, so I ended up just converting the weights myself. For ease of conversion, all of my Equinox fields needed to match their corresponding modules in the BFL implementation. This is easy to do in general, just loop over the safetensors and turn them into pytrees then just load them into the `Equinox` tree. One exception to this is the `torch.nn.Sequential` at the end of the single DiT blocks, which doesn't naturally map to pytrees. This was circumvented by just hardcoding the case into the weight loader. Not perfect solution but it's code that only needs to run once, so its fine.
Currently we load the weights for the VAE T5 and CLIP from Hf and shuffle data around to get it into jax arrays. I intentionally chose not to reimplement these in `JAX`, since it would derail me from actually working on the actual Flux code. I'll eventually move it all over to raw JAX/Equinox. This would make the code a lot cleaner, and would remove all of the torch dependencies. 

## Samples

<div class="sample-grid">
  <img src="posts/jlux/out_0.png" alt="Generated sample: a warmly lit corner bookstore at dusk" />
  <img src="posts/jlux/out_1.png" alt="Generated sample: a ginger cat sitting on a windowsill" />
  <img src="posts/jlux/out_2.png" alt="Generated sample: a foggy forest road at dawn" />
</div>

## Why is Flux.1 so Good?
There's no secret sauce. A simple objective, joint attention, and a lot of blocks. Rectified flow earns its keep by being simple to scale. 

## Loose Ends
- Need to reimplement Flux.1 Schnell as well. The architecture is basically identical, its just timestep distilled and doesn't take a guidance parameter.
- T5, vae, clip in JAX or use existing implementations. 
- Support all current open BFL models. Needs a refactor to handle different types of modules, but they share many components so it shouldn't be too bad
- Make it faster
- Add LoRA training capabilities

<a class="repo-badge" href="https://github.com/3xela/jlux"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg><span>3xela/jlux</span></a>

[^sdxl]: Podell et al., *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*, [arXiv:2307.01952](https://arxiv.org/abs/2307.01952)
[^flashattn]: Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
[^gaoflow]: Gao et al., *Diffusion Models and Gaussian Flow Matching: Two Sides of the Same Coin*, The Fourth Blogpost Track at ICLR 2025, [openreview](https://openreview.net/forum?id=C8Yyg9wy0s)
[^rectflow]: Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
[^flux]: Black Forest Labs, *FLUX.1*, [github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux)