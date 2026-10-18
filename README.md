# ComfyUI-Anima-DAVE

**DAVE** (DC Attenuation for diVersity Enhancement, training-free) as a one-node
ComfyUI **MODEL patch** for the Anima / Cosmos-`predict2` DiT and for
**MOSS-Image** (the [`moss-image`](https://github.com/OpenMOSS/MOSS-Image) node).

Diffusion models often collapse to near-identical layouts across seeds for the
same prompt. DAVE recovers that diversity by **attenuating the DC component** —
the per-channel spatial average `μ` — of each *early-mid* transformer block's
output, over only the **first `tau` fraction** of denoising steps. The DC carries
the seed-shared global layout/conditioning; the AC residual `h − μ` carries the
seed-specific structure. Pulling DC down lets the AC breathe without rewriting it:

```
ĥ = α·μ + (h − μ) = h − (1 − α)·μ        with  (1 − α) = strength · w(ℓ)
```

## Install

Clone into `ComfyUI/custom_nodes/` (or install via ComfyUI-Manager):

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sorryhyun/ComfyUI-Anima-DAVE comfyui-anima-dave
```

No extra dependencies (torch + numpy ship with ComfyUI). The block-pool masks are
bundled (`dave_alpha.npz` for Anima, `moss_dave_alpha.npz` for MOSS-Image).

## Usage

Insert **Anima / MOSS DAVE (after compile)** between your model loader and the KSampler:

```
UNETLoader ─▶ Anima DAVE ─▶ KSampler ─▶ …
```

If you also use **Anima Block Compile**, DAVE must come **after** it:

```
UNETLoader ─▶ Anima Block Compile ─▶ Anima DAVE ─▶ KSampler ─▶ …
```

Run a seed grid and you'll see same-prompt diversity open up.

### Knobs

| Input | Default | Notes |
|-------|---------|-------|
| `mask` | `auto` | Per-block pool `w(ℓ)`. `auto` picks the shipped mask for the loaded model: flat **blocks 8–18** on Anima (`dave_alpha.npz`), **blocks 19–30** of 42 on MOSS-Image (`moss_dave_alpha.npz`). |
| `strength` | `0.30` | Per-block DC-removal dose `s = (1−α) = strength·w(ℓ)` — **not a linear "more = stronger" knob** ([see below](#what-strength-actually-does-issue-1)). Its effect on *structure* is non-monotonic and often peaks **low** (`~0.05–0.2`); `0.30` is a safe all-round default, `~0.80` removes the most DC at `tau=0.10`. `0` disables. |
| `tau` | `0.06` | Fraction of early (high-σ) steps DAVE is active (3 of 50 steps). **Keep ≤ 0.10.** `0` = every step. |

### The one rule that matters

**Window width hurts more than dose.** Text and hands stay legible at `tau=0.10`
even at `strength=0.80`, but at `tau=0.15` even `strength=0.50` starts garbling
them. Tighten `tau` first; spend any remaining headroom on `strength`.

Recommended starting points:

- **Safe / default:** `tau=0.06`, `strength=0.30`
- **Most layout diversity:** `tau=0.10`, sweep `strength` **low** first (`0.05 → 0.2`)
- **Most raw DC removed:** `tau=0.10`, `strength=0.80`
- **Avoid:** `tau ≥ 0.15` (legibility falls off a cliff)

### What `strength` actually does (issue #1)

`strength` is **not** a global "more = stronger" intensity multiplier. It sets the
per-block DC-removal dose

```
(1 − α_ℓ) = strength · w(ℓ)        # fraction of DC removed at pooled block ℓ
h ← h − (1 − α_ℓ) · μ              # the edit, applied per pooled block
```

where `w(ℓ)` is the block's pool weight — ranked offline by
`power_ratio(ℓ) · max(0, DC−AC gap(ℓ))` and binarized in the shipped mask to a flat
`1.0` over the **early-mid structural band, blocks 8–18** (later detail blocks are
deliberately excluded).

Why the diversity effect peaks **low** rather than at the top of the range: those
pooled blocks set the **overall structure**, so even a small dose there reshuffles
same-prompt *layout* — the change you actually want. Pushing `strength` higher
removes more DC (more total change) but trades structural diversity for legibility,
and the network increasingly compensates, so layouts can converge again while text
and hands degrade. So for fresh layouts, **sweep low strengths first** (`~0.05–0.2`);
reach for `~0.80` only when you want the maximum DC knocked out and have `tau` pinned
at `0.10`.

> With a **custom continuous mask** (non-binary `w(ℓ)`), `strength` additionally
> acts as a soft selector: lowering it drops the lowest-weight blocks below the
> activation threshold first, so the dose concentrates on the highest-weight
> (structure-carrying) blocks — the same low-strength sweet spot, by a second route.

## MOSS-Image

Wire it after the model-side nodes of the `moss-image` example workflow:

```
MOSS-Image HF Loader ─▶ ModelSamplingSD3 ─▶ SkipLayerGuidanceDiTSimple (22) ─▶ DAVE ─▶ KSampler
```

`mask=auto`, `strength=0.30`, `tau=0.06` (the defaults) are the MOSS operating
point too. Differences from Anima, all handled inside the node:

- **Image tokens only.** A MOSS block outputs `[image | text]`; μ is the mean over
  the image tokens and the text tokens pass through.
- **Massive channels left out.** On MOSS ~99 % of the image tokens' mean sits in
  a handful of massive-activation channels (channel 1356 alone is ~1900× the median
  at block 0) that are identical across seeds and prompts. Removing a fraction of
  them shifts the block output instead of unlocking it, so the 18 channels listed
  in the mask's `exclude_channels` are left out of μ.
- **Pool 19–30.** Measured on MOSS-Image-Base (3 prompts × 8 seeds, cross-seed
  similarity of the seed-specific residual at the last step): blocks 8–18 alone
  move it 0.406 → 0.379; 19–30 move it to 0.209, the same as 8–30; adding 31–40
  gains little and makes the per-seed global tone drift further.
- **Expect** more varied framing / pose / panel layout, legible text, and slightly
  looser hands and objects than vanilla; now and then a global attribute (e.g. a
  sunset sky) drifts on one seed.

## How it works

DAVE edits *per-block intermediate features* gated by *per-step σ*, so it has to
live inside the model forward — hence a MODEL patch, not a sampler or latent node.
The patch installs a keyed `APPLY_MODEL` wrapper that, on each step, reads the
live σ schedule from `transformer_options` to decide the `tau` gate, binds the
live `diffusion_model` off the executor, and (when active) registers a
post-`forward` hook on each pooled block that subtracts `(1−α)·μ`. Hooks are torn
down after every `apply_model` call, so nothing leaks across seeds or runs.

## Composition / caveats

- **Compose order with Block Compile:** wire DAVE **after** Anima Block Compile
  (`loader → Block Compile → DAVE → sampler`). Block Compile swaps each block for
  a compiled `OptimizedModule` *inside* its own `APPLY_MODEL` wrapper; patcher
  wrappers nest in wiring order, so DAVE-after runs *inside* that swap and hooks
  the live compiled modules (the hook fires in eager after the compiled forward).
  Wiring DAVE *before* Block Compile puts it outside the swap and it silently
  no-ops (fails safe — no corruption).
- **CFG:** the edit applies uniformly to both the cond and uncond forwards.
- **Custom samplers:** the `tau` gate maps the current step against
  `transformer_options["sample_sigmas"]`. A sampler that doesn't publish the
  schedule falls back to running DAVE on every step (same as `tau=0`).

This is a faithful port of the in-repo `library/inference/corrections/dave.py`
(`--dave` CLI path); the DC-mean math is bit-for-bit identical on ComfyUI's
native 5D `(B,T,H,W,D)` block output. The MOSS path matches the trainer repo's
`library/moss/dave.py` bit-for-bit on a block output.

## Reference

DAVE is introduced in:

> **Breaking the Lock-in: Diversifying Text-to-Image Generation via
> Representation Modulation**, ICML 2026.

- Paper repo: <https://github.com/daheekwon/DAVE>

## License

MIT (this node). DAVE method © its authors.
