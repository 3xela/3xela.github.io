# Image Blocked by Safety Filter

Give the open weights of Ideogram 4 a vague prompt such as "a cat", "a concert poster", or "a landscape", generate a few dozen images, and the outputs barely vary. Most outputs are the same gray card reading *"Image blocked by safety filter."* Across vague prompts and every seed, that card appears about 60% of the time. This post explains why an image model answers a short prompt with a refusal placeholder.

The explanation runs from the output level down to a single attention head, fourteen blocks into the network, that decides whether a generation becomes a picture or the card.

Every result here is measured on the open `ideogram-ai/ideogram-4-nf4` weights [10], on an A100, with a 20-step sampler at 1024² and fixed seeds. None of the results come from the hosted product. This is a curiosity-driven teardown, not a safety claim.

![montage of 29 identical gray cards and one lone headshot](posts/ideogram-4/fig4_montage_seed0.png)

## Most of the convergence is one repeated image

The first step was to make the anecdote quantitative. The test set is 30 vague prompts plus 30 rich captions of the same subjects, at 6 seeds each, for 360 images. For each condition, DINOv2 [1] distance is measured two ways: *between* different prompts at the same seed, and *within* one prompt across seeds. The within-prompt distance is the seed-noise floor. The exact formulas are in the appendix.

| condition | between-prompt | within-prompt | ratio |
|---|---|---|---|
| vague — all images | 0.351 | 0.560 | **0.63** |
| vague — true renders only | 0.754 | 0.708 | 1.06 |
| dense — true renders only | 0.921 | 0.305 | **3.02** |

The ratio of 0.63 carries the result. A ratio below 1 means two different prompts land closer together than one prompt does across seeds. That outcome is only possible when most of the "different" outputs are the same image, which here is the card.

The second row matters as much as the first. Vague prompts that do render reach a ratio of 1.06, which is indistinguishable from seed noise, so rendered vague prompts are not collapsing into a single scene. The "all vague prompts look the same" effect is therefore not a drift in visual style. The effect is a discrete split between rendering and falling into the card. The remaining question is why an image model responds to a thin prompt by drawing a refusal.

## The card is a trained output, not a failure to draw

The card contains typeset English text. A model that fails to draw produces blur, whereas a specific image carrying legible words is something the model was trained to produce. The card is therefore a supervised training target.

The card also requires the prompt to trigger it. Turning text conditioning off entirely, by setting the classifier-free-guidance [4] weight to 0, produces a generic photo rather than the card, so gray is not the model's default output. The card appears specifically when a thin prompt is read. Every card is also the same card: averaging the final latents of eleven different carded prompts and decoding the mean produces a still-legible "Image blocked" card. The card occupies a single tight point in latent space, far from the manifold of real images.

The trigger is the amount of specification in the prompt. Raw richness drives the trigger, rather than JSON or any particular structure. Specification amount is readable from the prompt's text features before any denoising, at about 0.76 AUC, and it is the dominant readable axis.

## The decision is made early and is readable early

Sampling runs for 20 steps, but the choice between card and render is fixed within the first two. Running prompt A's conditioning for the first two steps and then switching to prompt B leaves the outcome set by prompt A. This swap test covers three subjects at one seed. The crossover is sharp and identical for all three, but the exact step is indicative rather than a calibrated constant.

A cheaper readout exists. Ideogram 4 is a flow-matching model [2, 3], so its first velocity prediction already points toward its destination. Running the network once at the initial noise and checking whether that velocity points at the card's latent separates card-bound from render-bound generations at 0.995 AUC. This figure comes from 60 prompts at one seed, 52 card and 8 render. The separation is large, but the small number of renders and the single seed mean it is not yet stress-tested. The model indicates the refusal before it has drawn anything.

A cheap detector was not the goal; locating the mechanism was.

## Where the decision lives: blocks 10–14

The generator is a 34-block diffusion transformer [5]. Two interventions locate where the verdict is written.

The first intervention is activation patching [6, 7]: splice a "will render" computation into a "will card" computation at block *l*, then let the rest of the network run. Splicing at block 9 or earlier leaves the receiver's outcome intact, because the patch is overwritten downstream. Splicing at blocks 12–14 makes the donor's outcome win. The decision is computed in blocks 10–14, and the remaining twenty blocks execute it. This localization rests on one-to-two prompt pairs at a single seed, and blocks 10–12 were not sampled, so "by 14" is firm while the lower edge of the band could be as early as 10.

The second intervention is attention severing: cut the image tokens' ability to read the prompt within a chosen band of blocks, then observe the output. The severing table below is a single-example dissection using one vague and one dense prompt at one seed, and is the part of the study most in need of replication.

| where we cut text-reading | vague prompt | dense prompt |
|---|---|---|
| nowhere | card | renders |
| blocks 0–9 | renders | renders |
| blocks 10–14 | card | cards |
| everywhere | renders (text-blind default) | renders (same default) |

The table holds the central result. Cutting the early text read stops the card from firing, and the vague prompt renders. Cutting the middle text read forces the card on every prompt, however rich. Cutting all text reading produces no card at all, only the model's generic text-blind image.

Thin prompts therefore do not summon the card directly. Every prompt arms the card through the early read in blocks 0–9. A sufficiently specified prompt then cancels the card by passing a content check in blocks 10–14. The card is the default outcome when that cancellation fails.

## One attention head carries the verdict

Each block has 18 attention heads and one MLP. Sublayer patching shows that attention, not the MLPs, moves the decision, and within attention the signal concentrates on one head: block 14, head 9.

A one-head result from a single example is not trustworthy on its own, so head (14, 9) was tested three ways, on prompts that normally render, using a hand-audited card detector.

The first test asks whether the head is general or cherry-picked. Averaging the card-ward push from patching each head over 18 prompt pairs, one cell stands out.

![per-head heatmap, block 14 head 9 dominant](posts/ideogram-4/fig31_general.png)

| head | mean push toward card | times it was top head |
|---|---|---|
| **(14, 9)** | **0.178** | **8 / 18** |
| (14, 17) | 0.113 | 5 / 18 |
| (13, 2) | 0.091 | 4 / 18 |
| everything else | ≤ 0.03 | 1 / 18 |

The second test asks whether the head is necessary. Ablating head (14, 9) [8] on 8 prompts that render fine, across 2 seeds (16 images per row below), raises the refusal rate sharply. The control matters here, because in this model almost any large perturbation causes a card, so the comparison is against ablating other heads the same way.

| ablate this head | refusal rate |
|---|---|
| nothing (baseline) | 6% |
| **(14, 9)** | **69%** |
| control (14, 5) | 12% |
| control (14, 17) | 19% |
| control (9, 9) | 6% |

Ablating one head turns two-thirds of normal renders into the card, which is 4–11× the rate of any matched control.

![ablation gallery: (14,9) columns are cards, controls still render](posts/ideogram-4/fig32_ablate.png)

The third test asks whether the head is sufficient. Transplanting a vague prompt's (14, 9) output into a dense prompt's generation, across 8 dense receivers, 2 seeds, and one vague donor, raises the refusal rate from 6% to 69%. One head's signal, pasted in, turns a render into the card.

![impose gallery: dense prompts carded by one transplanted head](posts/ideogram-4/fig33_impose.png)

Generality, necessity, and sufficiency point to the same conclusion. Head 9 of block 14 carries the card-versus-render decision.

## The gate is one-sided

Head (14, 9) is the dominant gate, not the only one. Heads (14, 17) and (13, 2) carry a minority of the signal, and about 30% of prompts resist a single-head push. The gate is a small set of heads with one main head, rather than a single off-switch neuron.

The gate is also asymmetric. Imposing the card is a one-head operation. Rescuing a carded prompt is not concentrated in any head, requires a broad distributed correction, and even then tends to produce a generic dark glyph rather than the requested subject. The geometry behind the asymmetry is straightforward: the card is a broad sink that almost any perturbation falls into, while reaching a specific coherent image is a constrained operation that a rich prompt performs. Head (14, 9) is the trip-wire into the card, not the route back out.

## What the card region looks like

The previous section asserted a geometry, so the geometry was measured. Embedding all 360 images with DINOv2 and viewing the cloud from the card's position shows the cards as a thin spike, a roughly 20° cone around a near-degenerate point, while the real renders spread over a roughly 80° fan. The card region is one tight sink surrounded by a wide space of other images.

The sink also forms visibly during sampling. Capturing the latent at all 20 steps for 60 prompts and tracking each cluster's distance to the card point shows the card-bound cluster contracting onto the point (RMS 866 → 174) while the render-bound cluster moves away (866 → 881). The basin closes on the card-bound prompts and opens for the rest.

![per-step snapshots: the card-bound cluster collapses onto the apex, the render cluster fans out](posts/ideogram-4/fig40_cluster_contraction.png)

The first expectation that proved wrong was a clean modality gap. The modality gap is the CLIP phenomenon [11] in which two kinds of representation occupy two separate narrow cones with empty space between them. A clean separation between cards and renders would have been simple to describe, but the data shows no such gap. The card cone has a long graded tail that overlaps render territory: the most render-like card sits at cosine 0.17 from the card point, and the most card-like render sits at cosine 0.69, so the two populations overlap. The overlap is filled by the card family described earlier, including a navy-blue "Image blocked" card partway out and a refusal-text-over-a-real-poster hybrid further out. The refusal target does not end at a sharp boundary, but fades into real images by degrees. The card region is a spike with a populated slope rather than an isolated point.

## Climbing back out of the card basin

A basin invites a direct question: can a card-bound trajectory be pushed out of it during sampling? The test took prompts headed for the card and, at a single denoising step, added one push to the latent along the direction from the card point toward the renders, leaving the prompt unchanged.

The second expectation that proved wrong was a latent point of no return. The conditioning swap locks the outcome by step 2, so a late latent push was expected to fail. Instead, a one-shot push escapes the card at every step tested, including step 12 of 20: all 18 pushed runs left the card, while the un-pushed baselines stayed. The latent is never trapped, and can always be pushed out of the sink. The point of no return found earlier is a property of the conditioning, not of the latent.

The content of the escaped image is the important catch. Escaping the card is not the same as drawing the prompt. The prompt "a cat", pushed gently and early, escapes the card and renders a detailed portrait of a man. Pushing harder or later degrades the output to a smeared figure and then to off-manifold noise. The push encodes the direction out of the card but encodes nothing about cats, because only the conditioning carries the subject, and the conditioning was left unchanged.

![a cat prompt pushed out of the card: escaped, but a man, then degraded, then noise](posts/ideogram-4/fig43_nudge_gallery.png)

A generation can be pulled out of the refusal sink at will, and still will not be the requested image. Head (14, 9) is the trip-wire into the card, and the conditioning is the only route to the requested picture.

## An unexplained case: dense portraits

The refusal is not limited to thin prompts. Richly specified portraits of people refuse more often than vague prompts, at 77%, and that decision is invisible to the text-feature probe that predicts the other cases. A second refusal pathway is the likely explanation. Whether that pathway runs through head (14, 9) or a different head is unknown, and a different head would mean this model has two gates and this post has found one.

## Reproducing this

The study uses open weights, fixed seeds, and a set of small scripts: generation and metrics, the mechanistic probes (`commit.py`, `ditprobe.py`, `gateprobe.py`, `gatesci.py`), and the geometry and escape pass (`conegap.py` for the cone, `conetraj.py` for the latent trajectories, `nudge.py` for the one-shot push). One methodological note is worth carrying over. The obvious blur-based card detector mislabels about 1 image in 9, because the card comes in a family of colored and garbled variants, so every number above uses labels from a pixel-level text matcher hand-audited against all 360 images.

In one sentence: an underspecified prompt arms a learned refusal in the first third of the network, and when the prompt cannot pass a content check around block 14, one attention head delivers the verdict and the model draws the "blocked" card instead of the requested image.

## Appendix: metrics as computed

**Image distance.** Each image $x$ receives an $\ell_2$-normalized CLS-pooled DINOv2 ViT-L/14 embedding $e(x)$ [1]. Distance between two images is cosine distance:

$$d(x, y) = 1 - \langle e(x),\, e(y) \rangle$$

**Between/within ratio.** Let $x_{p,s}$ be the image for prompt $p$ at seed $s$. The between-prompt distance averages pairwise distances across prompts at a shared seed, and the within-prompt distance averages across seeds of a shared prompt:

$$\mathrm{between} = \mathop{\mathbb{E}}_{s}\; \mathop{\mathbb{E}}_{p \neq q}\, d(x_{p,s},\, x_{q,s}), \qquad \mathrm{within} = \mathop{\mathbb{E}}_{p}\; \mathop{\mathbb{E}}_{s \neq s'}\, d(x_{p,s},\, x_{p,s'}), \qquad R = \frac{\mathrm{between}}{\mathrm{within}}$$

A ratio $R \approx 1$ means different prompts are no more distinguishable than seed noise. A ratio $R < 1$ is only possible when many "different" outputs are the same image.

**Card detector (audited).** Each image is converted to grayscale at 512². A horizontal band $B$ is binarized by robust deviation from its median, $m(B) = \mathbb{1}\left[\,|B - \mathrm{med}(B)| > 6 \cdot \mathrm{MAD}(B)\,\right]$. The best IoU against the canonical card's text band is then taken over a sliding 27-row window:

$$\mathrm{iou}(g) = \max_{200 \le y < 285} \frac{|m(g_{y:y+27}) \cap m_{\mathrm{tpl}}|}{|m(g_{y:y+27}) \cup m_{\mathrm{tpl}}|}, \qquad \text{card} \iff \mathrm{iou} \ge 0.33$$

This detector catches the colored and garbled card variants. The blur heuristic it replaced, the variance of the 4-neighbour Laplacian $\mathrm{Var}(\nabla^2 g) < 50$, mislabels 40 of 360 images and is used nowhere in this post.

**Guided velocity.** The classifier-free-guidance combination [4] of the model's conditional and unconditional velocity predictions is $v = w\, v_{\mathrm{cond}} + (1 - w)\, v_{\mathrm{uncond}}$. The main 360-image run uses the production preset's ramped schedule, with $w = 3$ for the first two steps and $7$ thereafter. The mechanistic probes hold $w = 7$ constant so trajectories are comparable across prompts.

**Step-0 card score.** Let $z_0$ be the initial noise, $v_0$ the guided velocity there, and $\bar{g}$ the mean final latent of card-bound runs:

$$s_0 = \cos\!\left(v_0,\; \bar{g} - z_0\right)$$

Thresholding $s_0$ separates card-bound from render-bound at 0.995 AUC in one forward pass with no sampling, measured on 60 single-seed prompts (52 card and 8 render). The mean $\bar{g}$ is built from that same card set, so the figure is an upper bound. The head-patching heatmap reports shifts of this score, $\Delta = s_0^{\mathrm{patched}} - s_0^{\mathrm{baseline}}$.

**Text-feature probe.** The DiT conditions on Qwen3-VL [9] hidden states stacked from 13 tap layers (13 × 4096 = 53,248 dims), mean-pooled over real text tokens to give one vector $c \in \mathbb{R}^{53248}$ per prompt. The probe is the difference of class means, $w = \mu_{\mathrm{card}} - \mu_{\mathrm{render}}$, scored as $w^{\top} c$, with $w$ refit leave-one-out so it never sees the test prompt. This probe gives the 0.76 AUC quoted above.

**Head interventions.** Each attention output is 18 heads × 256 dims entering the output projection. Patching replaces head $(b, h)$'s 256-dim slice with the donor prompt's slice at the identical $(z, t)$. Ablation either zeroes the slice or sets it to its mean, and both give the same card rates, so the signal matters rather than the magnitude. Refusal rates are the fraction of full 20-step samples the audited detector classifies as cards.

**AUC.** AUC throughout is the area under the ROC curve, which equals the probability that a randomly chosen card-bound example scores above a randomly chosen render-bound one (1.0 is perfect separation, 0.5 is chance).

**Cone geometry.** Each image's unit DINOv2 embedding is scored against the card point $\hat{g} = \mathrm{unit}(\mathrm{mean}_{\mathrm{card}}\, e)$. The card and render cone half-angles are the mean of $\arccos(\langle e, \hat g\rangle)$ over each class, 20° and 80° respectively, and both are robust to taking $\hat g$ as the median of pure cards instead. "No clean gap" means the per-class ranges of $\langle e, \hat g\rangle$ overlap, with a card floor of 0.17 below a render ceiling of 0.69. The latent cluster contraction is the per-step RMS distance of each class's captured latents to the card point (the mean final card latent), over 60 seed-0 trajectories.

**Latent nudge.** For a card-bound prompt, one fixed vector $m\cdot\mathrm{gap}\cdot\hat u_1$ is added to the latent $z$ after a single chosen step, where $\hat u_1 = \mathrm{unit}(\text{render-final-centroid} - \text{card point})$ is the escape direction and $\mathrm{gap}$ is its render-side coordinate. The conditioning is left unchanged. Escape means the audited card detector returns false on the final image. The test covers 3 card-bound prompts × steps {1, 2, 4, 8, 12} at a single seed, and content was checked by eye rather than by the detector alone. Two stated expectations failed: the clean two-cone modality gap is not observed, because the card family bridges it; and the expected latent point of no return is not observed, because the latent escapes at every step, which places the early lock-in in the conditioning alone.

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

[11] Liang et al., *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning*, NeurIPS 2022. [arXiv:2203.02053](https://arxiv.org/abs/2203.02053)

<!-- Publishing note — figures to copy from research/figures/:
  fig4_montage_seed0.png (hook), fig31_general.png, fig32_ablate.png,
  fig33_impose.png, fig40_cluster_contraction.png, fig43_nudge_gallery.png.
  Images sit flat in posts/ideogram-4/ and inline paths are posts/ideogram-4/<file>. -->
