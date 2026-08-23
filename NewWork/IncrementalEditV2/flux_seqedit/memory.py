"""
Step 5 — Layer-wise memory + generation-order occlusion.

Design points baked in from the spec:
  * Store the RAW locked mask per object, never its currently-visible version.
    Visible masks are recomputed at use-time so a late edit can correctly
    re-occlude an early one.
  * Occlusion is generation-order: newest on top. Visible mask of layer j is the
    cascading subtraction  m_j - union(m_{j+1..N}).
  * M_memory (for the next edit's repulsion prior) is the union of raw masks.

Masks here are 2D boolean patch-grid arrays (ph, pw) — the resolution the
attention signal lives at. Everything is numpy; no torch needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class EditLayer:
    prompt: str
    raw_mask: np.ndarray            # bool (ph, pw), the object's own extent
    token_span: tuple[int, int]     # (start, end) indices into text tokens
    meta: dict = field(default_factory=dict)


class LayerMemory:
    """Ordered store of edit layers. Index 0 is the background (full-frame)."""

    def __init__(self, ph: int, pw: int):
        self.ph, self.pw = ph, pw
        self.layers: list[EditLayer] = []

    # ---- construction -------------------------------------------------------

    def set_background(self, prompt: str, token_span=(0, 0)):
        full = np.ones((self.ph, self.pw), dtype=bool)
        if self.layers:
            self.layers[0] = EditLayer(prompt, full, token_span, {"background": True})
        else:
            self.layers.append(EditLayer(prompt, full, token_span, {"background": True}))

    def add_layer(self, prompt: str, raw_mask: np.ndarray, token_span=(0, 0), **meta):
        m = np.asarray(raw_mask, dtype=bool)
        if m.shape != (self.ph, self.pw):
            raise ValueError(f"raw_mask shape {m.shape} != grid ({self.ph},{self.pw})")
        self.layers.append(EditLayer(prompt, m, token_span, dict(meta)))
        return len(self.layers) - 1

    # ---- occlusion ----------------------------------------------------------

    def occupied_union(self, exclude_background: bool = True) -> np.ndarray:
        """Union of raw masks of all *object* layers -> M_memory for the prior."""
        acc = np.zeros((self.ph, self.pw), dtype=bool)
        for i, layer in enumerate(self.layers):
            if exclude_background and layer.meta.get("background"):
                continue
            acc |= layer.raw_mask
        return acc

    def visible_mask(self, j: int) -> np.ndarray:
        """
        Visible extent of layer j under generation-order occlusion:
            m_j minus the union of all later object layers.
        Background (j=0) is visible wherever no object covers it.
        """
        if not (0 <= j < len(self.layers)):
            raise IndexError(j)
        base = self.layers[j].raw_mask.copy()
        later = np.zeros((self.ph, self.pw), dtype=bool)
        for k in range(j + 1, len(self.layers)):
            later |= self.layers[k].raw_mask
        return base & ~later

    def all_visible_masks(self) -> list[np.ndarray]:
        return [self.visible_mask(j) for j in range(len(self.layers))]

    # ---- introspection ------------------------------------------------------

    def __len__(self):
        return len(self.layers)

    def summary(self) -> str:
        rows = []
        for j, l in enumerate(self.layers):
            vis = self.visible_mask(j)
            rows.append(f"[{j}] '{l.prompt}' raw={int(l.raw_mask.sum())} "
                        f"visible={int(vis.sum())} span={l.token_span}")
        return "\n".join(rows)
