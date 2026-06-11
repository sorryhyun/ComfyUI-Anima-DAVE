"""Anima DAVE — DC Attenuation for diVersity Enhancement, as a ComfyUI MODEL patch.

DAVE (training-free, ICML'26) recovers same-prompt sample diversity by attenuating
the **DC component** of each target Transformer block's output — the per-channel
spatial average ``μ^ℓ`` that is near-perfectly cross-seed-shared (it carries the
conditioning / global layout) while the **AC residual** ``h − μ`` holds the
seed-specific structure. The per-block edit::

    ĥ^ℓ = α_ℓ · μ^ℓ + (h^ℓ − μ^ℓ) = h^ℓ − (1 − α_ℓ) · μ^ℓ      (α_ℓ ≤ 1)

lets the diverse AC breathe without rewriting it. ``(1 − α_ℓ) = strength · w(ℓ)``
where ``w(ℓ) ∈ {0,1}`` is the shipped offline-derived pool mask (``dave_alpha.npz``,
flat blocks 8–18). The edit is additionally **σ-gated** to the first ``tau``
fraction of denoising steps.

Why a MODEL patch (and not a sampler / latent node): DAVE edits *per-block
intermediate features* gated by *per-step σ*. Neither is reachable from a sampler
or latent node — it must live inside the model forward. We install it via
``set_model_unet_function_wrapper``: on each step the wrapper reads the live σ
schedule from ``transformer_options`` to decide the gate, then (when active)
registers a post-``forward`` hook on each pooled DiT block, runs ``apply_model``,
and removes the hooks in a ``finally`` so nothing leaks across runs/seeds. This
mirrors the in-repo ``library/inference/corrections/dave.py`` math bit-for-bit;
ComfyUI's native Cosmos/predict2 ``Block.forward`` returns a 5D ``(B,T,H,W,D)``
tensor, so the DC mean is over the spatial/token dims (all but batch and channel).

Caveat: composition with a torch.compile node (e.g. AnimaBlockCompile) is
**unverified** — block modules are swapped for compiled ``OptimizedModule``s
*inside* ``apply_model``, after this wrapper installs its hooks, so under compile
DAVE may silently no-op (fails safe, no corruption). Apply DAVE on the eager model,
or verify the compiled path on a live run before relying on it.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

logger = logging.getLogger("ComfyUI-Anima-DAVE")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MASK = "dave_alpha.npz"


def _list_masks() -> list[str]:
    """Bundled per-block weight masks (``*.npz`` with a ``weight`` array)."""
    masks = sorted(f for f in os.listdir(HERE) if f.endswith(".npz"))
    if DEFAULT_MASK in masks:  # surface the shipped mask first
        masks = [DEFAULT_MASK] + [m for m in masks if m != DEFAULT_MASK]
    return masks or [DEFAULT_MASK]


def _load_weight(mask_file: str) -> np.ndarray:
    """Load the per-block weight vector w(ℓ) ∈ [0, 1] from a bundled npz."""
    path = os.path.join(HERE, mask_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"DAVE mask not found: {path}")
    return np.load(path)["weight"].astype(np.float64)


def _dc_hook(atten: float):
    """Post-``forward`` hook: ``out ← out − atten · μ`` (μ = per-channel DC mean).

    The reduction keeps batch (dim 0) and channel (last dim); for ComfyUI's
    predict2 block output ``(B,T,H,W,D)`` that is dims ``(1,2,3)``. Generalized to
    ``range(1, ndim-1)`` so any 3D-token layout still reduces correctly.
    """

    def hook(_module, _inputs, output: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, output.ndim - 1)) or (1,)
        mu = output.float().mean(dim=dims, keepdim=True)
        return (output.float() - atten * mu).to(output.dtype)

    return hook


def _current_step(transformer_options: dict, timestep: torch.Tensor):
    """(step_index, n_steps) for the current forward, mapped against the schedule.

    Mirrors ComfyUI's own ``isclose(sample_sigmas, sigma_now)`` mapping
    (``comfy/context_windows.py``). Returns ``(None, None)`` if the schedule is
    unavailable (custom samplers that don't publish ``sample_sigmas``).
    """
    sched = transformer_options.get("sample_sigmas")
    if sched is None or len(sched) < 2:
        return None, None
    cur = transformer_options.get("sigmas", timestep)
    cur0 = cur.flatten()[0].to(sched.device)
    matches = torch.isclose(sched, cur0, rtol=1e-4, atol=1e-6)
    nz = torch.nonzero(matches).flatten()
    step = int(nz[0].item()) if nz.numel() else 0
    return step, len(sched) - 1


class AnimaDAVE:
    """Patch an Anima/Cosmos MODEL with DAVE same-prompt diversity enhancement."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mask": (
                    _list_masks(),
                    {"tooltip": "Per-block pool mask (w(ℓ)). Shipped: flat blocks 8–18."},
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.30,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "DC removal dose s = (1−α). 0.30 is the conservative default; "
                            "up to ~0.80 at tau=0.10 for max diversity. 0 disables."
                        ),
                    },
                ),
                "tau": (
                    "FLOAT",
                    {
                        "default": 0.10,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Fraction of the early (high-σ) steps DAVE is active. "
                            "KEEP ≤ 0.10 — wider windows garble text/hands far more than "
                            "extra dose does. Recommended 0.10. Set 0 to use the raw "
                            "sigma_lo/sigma_hi window instead."
                        ),
                    },
                ),
            },
            "optional": {
                "block_lo": (
                    "INT",
                    {"default": 0, "min": 0, "max": 255, "tooltip": "Lowest block DAVE may touch."},
                ),
                "block_hi": (
                    "INT",
                    {"default": -1, "min": -1, "max": 255, "tooltip": "Highest block (-1 = last)."},
                ),
                "sigma_lo": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Raw σ window low bound — used only when tau = 0."},
                ),
                "sigma_hi": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Raw σ window high bound — used only when tau = 0."},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "anima"
    DESCRIPTION = (
        "DAVE: recover same-prompt sample diversity by attenuating the DC component "
        "of early-mid DiT blocks over the first tau fraction of steps."
    )

    def patch(self, model, mask, strength, tau, block_lo=0, block_hi=-1, sigma_lo=0.0, sigma_hi=1.0):
        weight = _load_weight(mask)
        num_blocks = weight.shape[0]

        # Block-range cap: zero the mask outside [block_lo, block_hi] (hi=-1 → last).
        hi = num_blocks - 1 if block_hi < 0 else min(block_hi, num_blocks - 1)
        lo = max(0, block_lo)
        capped = weight.copy()
        if lo > 0:
            capped[:lo] = 0.0
        if hi < num_blocks - 1:
            capped[hi + 1 :] = 0.0

        # atten = (1 − α) = strength · w(ℓ), clamped so α stays in [0, 1].
        atten = np.clip(float(strength) * capped, 0.0, 1.0)
        pooled = [(i, float(a)) for i, a in enumerate(atten) if a > 1e-3]

        tau_f = float(tau)
        s_lo, s_hi = float(sigma_lo), float(sigma_hi)

        if not pooled:
            logger.info("DAVE: strength=0 or empty mask → no-op passthrough.")
            return (model,)

        active_idx = [i for i, _ in pooled]
        logger.info(
            "DAVE armed: strength=%.3f, tau=%.3f, %d/%d blocks active (%d..%d)",
            strength, tau_f, len(pooled), num_blocks, active_idx[0], active_idx[-1],
        )

        m = model.clone()

        def unet_wrapper(apply_model, args):
            x, t, c = args["input"], args["timestep"], args["c"]
            topts = c.get("transformer_options", {})

            # Decide the σ gate for this forward.
            if tau_f > 0.0:
                step, n_steps = _current_step(topts, t)
                if step is None:  # no schedule published → run every step (safe default)
                    gate = True
                else:
                    k = max(1, min(n_steps, round(tau_f * n_steps)))
                    gate = step < k
            else:
                cur = topts.get("sigmas", t)
                cur0 = float(cur.flatten()[0])
                gate = s_lo <= cur0 <= s_hi

            if not gate:
                return apply_model(x, t, **c)

            dm = m.get_model_object("diffusion_model")
            blocks = getattr(dm, "blocks", None)
            if blocks is None:
                return apply_model(x, t, **c)

            handles = [
                blocks[i].register_forward_hook(_dc_hook(a))
                for i, a in pooled
                if i < len(blocks)
            ]
            try:
                return apply_model(x, t, **c)
            finally:
                for h in handles:
                    h.remove()

        m.set_model_unet_function_wrapper(unet_wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {"AnimaDAVE": AnimaDAVE}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaDAVE": "Anima DAVE (Diversity)"}
