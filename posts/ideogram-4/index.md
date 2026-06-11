# We asked an image model for "a cat" 180 times. It refused — and we found the head that does it.

Give the open weights of Ideogram 4 a vague prompt — "a cat", "a concert poster", "a landscape" — generate a few dozen images, and you don't get variety. You get the same gray card, over and over: *"Image blocked by safety filter."* For vague prompts it happens about 60% of the time, across every seed. The model would rather draw a refusal than draw you a cat.

This post chases that quirk all the way down: from "the outputs all look the same" to the single attention head, fourteen blocks deep, that decides whether you get your picture or the card.

*(Everything here is measured on the open `ideogram-ai/ideogram-4-nf4` weights [10] — A100, 20-step sampler, 1024², fixed seeds — not the hosted product. A curiosity-driven teardown, not a safety claim.)*

![montage of 29 identical gray cards and one lone headshot](posts/ideogram-4/fig4_montage_seed0.png)

## "Convergence" is mostly one literal image

We started by making the anecdote quantitative: 30 vague prompts plus 30 rich captions of the same subjects, 6 seeds each, 360 images. For each condition, measure DINOv2 [1] distance *between* prompts at the same seed, and *within* one prompt across seeds — the seed-noise floor (exact formulas in the appendix).

| condition | between-prompt | within-prompt | ratio |
|---|---|---|---|
| vague — all images | 0.351 | 0.560 | **0.63** |
| vague — true renders only | 0.754 | 0.708 | 1.06 |
| dense — true renders only | 0.921 | 0.305 | **3.02** |

That 0.63 is the whole story. A ratio below 1 means two *different* prompts land closer together than the *same* prompt across seeds — which is only possible if most "different" outputs are the identical image. They are: the card.

The second row is just as important. Vague prompts that *do* render are not collapsing into one scene — their ratio of 1.06 is indistinguishable from seed noise. So the "all vague prompts look the same" effect is not style drift. It's a discrete coin flip: render, or fall into the card. The real question isn't why vague prompts converge in style. It's why an image model answers a thin prompt by drawing a refusal.

## The card is a trained behavior, not mush

The card has typeset English on it. A model failing to draw produces blur; a specific image with legible words is something the model learned to draw on purpose — a supervised training target.

And it needs the prompt to fire. Turn text conditioning off entirely (classifier-free guidance [4] weight 0) and you get a generic photo, not the card — gray is not where the model relaxes by default. The card appears precisely when a *thin* prompt is read. All the cards are also the *same* card: average the final latents of eleven different carded prompts and decode the mean, and out comes a still-legible "Image blocked" card. It's a single tight point in latent space, far from the image manifold.

The trigger is almost embarrassingly simple: how much you specified. Sheer richness — not JSON, not structure. It's readable off the prompt's text features before any denoising at about 0.76 AUC, and the dominant readable axis is specification amount.

## The decision is made early, and it's readable

Sampling takes 20 steps, but the card/render fate is locked in the first two. Swap the conditioning from prompt A to prompt B after step 2 and the swap no longer matters — whatever held steps 0–2 wins.

There's an even cheaper readout. This is a flow-matching model [2, 3], so its very first velocity prediction should lean toward its destination. Run the network *once* at the initial noise and check whether that velocity points at the card's latent: that single forward pass separates card-bound from render-bound at 0.995 AUC. The model announces the refusal before it has drawn anything.

A cheap detector is nice. But we wanted the mechanism.

## Where it lives: blocks 10–14, arm-then-cancel

The generator is a 34-block diffusion transformer [5], and two scalpels found where the verdict gets written.

The first is **activation patching** [6, 7]: splice a "will render" computation into a "will card" one at block *l* and let the rest of the network run. Splice in at block 9 or earlier and the receiver's outcome survives — the patch gets overwritten. By blocks 12–14 the donor's outcome wins. The decision is computed in blocks 10–14; the remaining twenty blocks just execute it.

The second is **attention severing**: cut the image tokens' ability to read the prompt inside a chosen band of blocks, and see what comes out.

| where we cut text-reading | vague prompt | dense prompt |
|---|---|---|
| nowhere | card | renders |
| blocks 0–9 | **renders!** | renders |
| blocks 10–14 | card | **cards!** |
| everywhere | renders (text-blind default) | renders (same default) |

Read the table twice — it's the strangest result in the study. Cut the *early* text read and the card never fires; a vague prompt happily renders. Cut the *middle* text read and everyone gets the card, no matter how rich the prompt. Cut everything and there's no card at all, just the model's generic text-blind image.

So thin prompts don't summon the card. *Every* prompt arms it, via that early read in blocks 0–9. A good prompt then earns its way out by passing a content check at blocks 10–14. The card is the default you fall into when the cancel fails.

## One attention head carries the verdict

Each block has 18 attention heads plus an MLP. Sublayer patching says attention, not the MLPs, moves the decision — and within attention the signal concentrates absurdly: one head, **block 14, head 9**.

A one-head result from one example is exactly the kind of thing you shouldn't believe, so we tested it three ways, on prompts that normally render, with a hand-audited card detector.

**Is it general, or cherry-picked?** Average the card-ward push from patching each head, over 18 prompt pairs. One cell lights up:

![per-head heatmap, block 14 head 9 dominant](posts/ideogram-4/fig31_general.png)

| head | mean push toward card | times it was top head |
|---|---|---|
| **(14, 9)** | **0.178** | **8 / 18** |
| (14, 17) | 0.113 | 5 / 18 |
| (13, 2) | 0.091 | 4 / 18 |
| everything else | ≤ 0.03 | 1 / 18 |

**Is it necessary?** Ablate just that head [8] on prompts that render fine. The control matters here: in this model almost *any* large perturbation causes a card, so the comparison is against ablating other heads the same way.

| ablate this head | refusal rate |
|---|---|
| nothing (baseline) | 6% |
| **(14, 9)** | **69%** |
| control (14, 5) | 12% |
| control (14, 17) | 19% |
| control (9, 9) | 6% |

Knock out one head and two-thirds of normal renders become the card — 4–11× more than any matched control.

![ablation gallery: (14,9) columns are cards, controls still render](posts/ideogram-4/fig32_ablate.png)

**Is it sufficient?** Transplant a vague prompt's (14,9) output into a dense prompt's generation, and the refusal rate goes from 6% to 69%. One head's signal, pasted in, flips a render into the card.

![impose gallery: dense prompts carded by one transplanted head](posts/ideogram-4/fig33_impose.png)

Generality, necessity, sufficiency — three independent measurements, one answer: head 9 of block 14 is the gate.

## Falling in is easy; climbing out is hard

The honest qualifier, which also turns out to be the interesting one: (14,9) is the *dominant* gate, not the only one. Heads (14,17) and (13,2) carry a minority of the signal, and about 30% of prompts shrug off a single-head push. It's a small head-set with one ringleader, not a literal off-switch neuron.

And there's a striking asymmetry. *Imposing* the card is a one-head chokepoint. *Rescuing* a carded prompt is not concentrated anywhere — it takes a broad, distributed correction, and even then the rescued output tends to be a generic dark glyph rather than your actual cat. The geometry explains why: the card is a broad sink, and almost any perturbation knocks you in, but climbing back out to a *specific, coherent image* is the constrained operation. That's the work a rich prompt does. The head is the trip-wire on the way down, not the ladder back up.

## The thread we haven't pulled

The refusal isn't only about thin prompts. Richly-specified *portraits of people* refuse **more** than vague ones — 77% — and that decision is invisible to the text-feature probe that predicts everything else. That looks like a second refusal pathway, and we don't yet know whether it runs through (14,9) or a different head. If it's a different one, there are two gates in this model and we've found one.

## Reproducing this

Open weights, fixed seeds, and a handful of small scripts: generation and metrics, then the mechanistic probes (`commit.py`, `ditprobe.py`, `gateprobe.py`, `gatesci.py`). One methodological note worth stealing: the obvious blur-based "is it the card" detector mislabels about 1 in 9 images, because the card comes in a family of colored and garbled variants. Every number above uses labels from a pixel-level text matcher, hand-audited against all 360 images.

The one-sentence version: an underspecified prompt arms a learned refusal in the first third of the network, and if it can't pass a content check around block 14, one attention head delivers the verdict and the model draws its "blocked" card instead of your image.

## Appendix: the metrics, exactly as computed

**Image distance.** Each image $x$ gets an $\ell_2$-normalized CLS-pooled DINOv2 ViT-L/14 embedding $e(x)$ [1]; distance is cosine:

$$d(x, y) = 1 - \langle e(x),\, e(y) \rangle$$

**Between/within ratio.** With $x_{p,s}$ the image for prompt $p$ at seed $s$, *between* averages pairwise distances across prompts at a shared seed, *within* across seeds of a shared prompt:

$$\mathrm{between} = \mathop{\mathbb{E}}_{s}\; \mathop{\mathbb{E}}_{p \neq q}\, d(x_{p,s},\, x_{q,s}), \qquad \mathrm{within} = \mathop{\mathbb{E}}_{p}\; \mathop{\mathbb{E}}_{s \neq s'}\, d(x_{p,s},\, x_{p,s'}), \qquad R = \frac{\mathrm{between}}{\mathrm{within}}$$

$R \approx 1$ means different prompts are no more distinguishable than seed noise; $R < 1$ is only possible when many "different" outputs are the same image.

**Card detector (audited).** Grayscale at 512², binarize a horizontal band $B$ by robust deviation from its median, $m(B) = \mathbb{1}\left[\,|B - \mathrm{med}(B)| > 6 \cdot \mathrm{MAD}(B)\,\right]$, then take the best IoU against the canonical card's text band over a sliding 27-row window:

$$\mathrm{iou}(g) = \max_{200 \le y < 285} \frac{|m(g_{y:y+27}) \cap m_{\mathrm{tpl}}|}{|m(g_{y:y+27}) \cup m_{\mathrm{tpl}}|}, \qquad \text{card} \iff \mathrm{iou} \ge 0.33$$

This catches the colored and garbled card variants. The blur heuristic it replaced — variance of the 4-neighbour Laplacian, $\mathrm{Var}(\nabla^2 g) < 50$ — mislabels 40 of 360 images and is used nowhere in this post.

**Guided velocity.** All sampling and probes use the standard classifier-free-guidance combination [4] of the model's conditional and unconditional velocity predictions, $v = w\, v_{\mathrm{cond}} + (1 - w)\, v_{\mathrm{uncond}}$ with $w = 7$.

**Step-0 card score.** With $z_0$ the initial noise, $v_0$ the guided velocity there, and $\bar{g}$ the mean final latent of card-bound runs:

$$s_0 = \cos\!\left(v_0,\; \bar{g} - z_0\right)$$

Thresholding $s_0$ separates card-bound from render-bound at 0.995 AUC — one forward pass, no sampling. The head-patching heatmap reports shifts of this score, $\Delta = s_0^{\mathrm{patched}} - s_0^{\mathrm{baseline}}$.

**Text-feature probe.** The DiT conditions on Qwen3-VL [9] hidden states stacked from 13 tap layers (13 × 4096 = 53,248 dims); mean-pool over real text tokens to get $c \in \mathbb{R}^{53248}$ per prompt. The probe is the difference of class means, $w = \mu_{\mathrm{card}} - \mu_{\mathrm{render}}$, scored as $w^{\top} c$ with $w$ refit leave-one-out so it never sees the test prompt. That's the 0.76 AUC quoted above.

**Head interventions.** Each attention output is 18 heads × 256 dims entering the output projection. Patching replaces head $(b, h)$'s 256-dim slice with the donor prompt's at the identical $(z, t)$; ablation zeroes it or sets it to its mean (both give the same card rates, so it's the signal that matters, not the magnitude). Refusal rates are fractions of full 20-step samples the audited detector classifies as cards.

**AUC** throughout is the area under the ROC curve: the probability a randomly chosen card-bound example scores above a randomly chosen render-bound one (1.0 = perfect separation, 0.5 = chance).

## References

[1] Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)

[2] Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

[3] Liu, Gong & Liu, *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, ICLR 2023. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)

[4] Ho & Salimans, *Classifier-Free Diffusion Guidance*, NeurIPS 2021 Workshop. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)

[5] Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)

[6] Meng et al., *Locating and Editing Factual Associations in GPT*, NeurIPS 2022. [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)

[7] Zhang & Nanda, *Towards Best Practices of Activation Patching in Language Models*, ICLR 2024. [arXiv:2309.16042](https://arxiv.org/abs/2309.16042)

[8] Michel, Levy & Neubig, *Are Sixteen Heads Really Better than One?*, NeurIPS 2019. [arXiv:1905.10650](https://arxiv.org/abs/1905.10650)

[9] Qwen Team, *Qwen3-VL*, 2025. [github.com/QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

[10] Ideogram, *ideogram-4-nf4* open weights. [huggingface.co/ideogram-ai/ideogram-4-nf4](https://huggingface.co/ideogram-ai/ideogram-4-nf4)

<!-- Publishing note — figures to copy from research/figures/:
  fig4_montage_seed0.png (hook), fig31_general.png, fig32_ablate.png,
  fig33_impose.png. Inline paths assume the images sit next to index.md. -->
