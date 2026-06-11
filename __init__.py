"""Anima DAVE ComfyUI custom node.

DAVE (DC Attenuation for diVersity Enhancement, training-free, ICML'26) as a
drop-in MODEL patch: it attenuates the DC component of early-mid DiT blocks over
the first ``tau`` fraction of denoising steps to recover same-prompt sample
diversity, leaving the seed-specific AC residual intact.

* ``AnimaDAVE`` — MODEL -> MODEL. Insert between the loader and any KSampler.

Self-contained: the per-block pool mask (``dave_alpha.npz``, flat blocks 8–18) is
bundled, and the ~6-line DC-removal math is reimplemented here, so the node pulls
no ``library.*`` / ``networks.*`` code from the training repo.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
