# ComfyUI-Anima-DAVE

**DAVE** (DC Attenuation for diVersity Enhancement, training-free) as a one-node
ComfyUI **MODEL patch** for the Anima / Cosmos-`predict2` DiT.

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

No extra dependencies (torch + numpy ship with ComfyUI). The block-pool mask is
bundled (`dave_alpha.npz`).

## Usage

Insert **Anima DAVE (Diversity)** between your model loader and the KSampler:

```
UNETLoader ─▶ Anima DAVE ─▶ KSampler ─▶ …
```

Run a seed grid and you'll see same-prompt diversity open up.

### Knobs

| Input | Default | Notes |
|-------|---------|-------|
| `mask` | `dave_alpha.npz` | Per-block pool `w(ℓ)`. Shipped mask is flat **blocks 8–18**. |
| `strength` | `0.30` | DC removal dose `s = (1−α)`. `0.30` conservative; up to `~0.80` at `tau=0.10` for max diversity. `0` disables. |
| `tau` | `0.10` | Fraction of early (high-σ) steps DAVE is active. **Keep ≤ 0.10.** |
| `block_lo` / `block_hi` | `0` / `-1` | Optional extra cap on the touched block range (`-1` = last). |
| `sigma_lo` / `sigma_hi` | `0.0` / `1.0` | Raw σ window — used **only when `tau = 0`**. |

### The one rule that matters

**Window width hurts more than dose.** Text and hands stay legible at `tau=0.10`
even at `strength=0.80`, but at `tau=0.15` even `strength=0.50` starts garbling
them. Tighten `tau` first; spend any remaining headroom on `strength`.

Recommended starting points:

- **Safe / default:** `tau=0.10`, `strength=0.30`
- **Max diversity:** `tau=0.10`, `strength=0.80`
- **Avoid:** `tau ≥ 0.15` (legibility falls off a cliff)

## How it works

DAVE edits *per-block intermediate features* gated by *per-step σ*, so it has to
live inside the model forward — hence a MODEL patch, not a sampler or latent node.
The patch installs a `set_model_unet_function_wrapper` that, on each step, reads
the live σ schedule from `transformer_options` to decide the `tau` gate, and (when
active) registers a post-`forward` hook on each pooled block that subtracts
`(1−α)·μ`. Hooks are torn down after every `apply_model` call, so nothing leaks
across seeds or runs.

## Composition / caveats

- **Compose order:** apply DAVE on the **eager** model. Composition with a
  `torch.compile` node (e.g. AnimaBlockCompile) is **unverified** — compiled
  block modules are swapped in *inside* `apply_model`, after this wrapper installs
  its hooks, so under compile DAVE may silently no-op (fails safe — no
  corruption). If you compile, verify a live run before relying on the effect.
- **CFG:** the edit applies uniformly to both the cond and uncond forwards.
- **Custom samplers:** the `tau` gate maps the current step against
  `transformer_options["sample_sigmas"]`. A sampler that doesn't publish the
  schedule falls back to running DAVE on every step — use the raw `sigma_lo` /
  `sigma_hi` window (`tau=0`) there.

This is a faithful port of the in-repo `library/inference/corrections/dave.py`
(`--dave` CLI path); the DC-mean math is bit-for-bit identical on ComfyUI's
native 5D `(B,T,H,W,D)` block output.

## License

MIT (this node). DAVE method © its authors.
