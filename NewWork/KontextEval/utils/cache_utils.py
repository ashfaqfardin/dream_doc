"""
Cache I/O: save K/V dictionaries to disk and reload them with shape verification.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch


def save_cache(store: dict, path: str, prefix: str = ""):
    """
    Save every tensor in `store` to `path/<prefix><key>.pt`.
    Non-tensor values are skipped.
    """
    os.makedirs(path, exist_ok=True)
    saved = []
    for key, val in store.items():
        if not isinstance(val, torch.Tensor):
            continue
        fname = os.path.join(path, f"{prefix}{key}.pt")
        torch.save(val.cpu(), fname)
        saved.append(key)
    return saved


def load_cache(path: str, prefix: str = "",
               keys: Optional[list[str]] = None,
               device: str = "cpu") -> dict:
    """
    Load tensors from `path`. If `keys` is given load only those,
    otherwise load everything matching `*.pt` in the directory.
    """
    p = Path(path)
    store = {}

    if keys is not None:
        for key in keys:
            fname = p / f"{prefix}{key}.pt"
            if fname.exists():
                store[key] = torch.load(fname, map_location=device)
    else:
        for fname in sorted(p.glob(f"{prefix}*.pt")):
            key = fname.stem
            if prefix:
                key = key[len(prefix):]
            store[key] = torch.load(fname, map_location=device)

    return store


def verify_cache(store_a: dict, store_b: dict,
                 rtol: float = 1e-3, atol: float = 1e-4) -> dict:
    """
    Compare two cache dicts.  Returns a summary dict with per-key statistics.
    """
    results = {}
    all_keys = set(store_a) | set(store_b)

    for key in sorted(all_keys):
        if key not in store_a:
            results[key] = {"status": "missing_in_a"}
            continue
        if key not in store_b:
            results[key] = {"status": "missing_in_b"}
            continue

        a = store_a[key].float()
        b = store_b[key].float()

        if a.shape != b.shape:
            results[key] = {"status": "shape_mismatch", "a": a.shape, "b": b.shape}
            continue

        max_diff = (a - b).abs().max().item()
        rel_diff = ((a - b).abs() / (a.abs() + 1e-8)).max().item()
        match = bool(torch.allclose(a, b, rtol=rtol, atol=atol))
        results[key] = {
            "status": "ok" if match else "mismatch",
            "shape": tuple(a.shape),
            "max_abs_diff": max_diff,
            "max_rel_diff": rel_diff,
        }

    return results


def cache_summary(store: dict) -> list[dict]:
    """Print/return a table of key → shape → dtype for every tensor in the store."""
    rows = []
    for key in sorted(store):
        val = store[key]
        if isinstance(val, torch.Tensor):
            rows.append({
                "key": key,
                "shape": tuple(val.shape),
                "dtype": str(val.dtype),
                "size_mb": val.element_size() * val.numel() / 1e6,
            })
    return rows
