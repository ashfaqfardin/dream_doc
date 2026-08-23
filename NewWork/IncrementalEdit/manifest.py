"""
Scene manifest — the incremental-editing backbone (pipelineInc.md §1).

Pure Python + JSON, no torch/numpy dependency, so this module is fully
unit-testable on CPU without a GPU (see test_cpu.py).

A project directory is self-contained:
    <project_dir>/manifest.json
    <project_dir>/canvas_v<N>.png
    <project_dir>/masks/<object_name>.png
    <project_dir>/runs/rev<N>_summary.json

Objects are addressable by name across separate process invocations run at
any later time — that is the actual "incremental" part of incremental
editing. Revisions form an append-only history; replacing/removing an object
retires it rather than deleting it.
"""

import json
import os
import tempfile
from typing import Dict, List, Optional


class ManifestError(Exception):
    pass


class SceneManifest:
    def __init__(self, project_dir: str, resolution: List[int],
                 revisions: Optional[List[dict]] = None,
                 objects: Optional[Dict[str, dict]] = None):
        self.project_dir = project_dir
        self.resolution = list(resolution)
        self.revisions: List[dict] = revisions if revisions is not None else []
        self.objects: Dict[str, dict] = objects if objects is not None else {}

    # ────────────────────────── persistence ──────────────────────────────

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.project_dir, "manifest.json")

    @classmethod
    def create(cls, project_dir: str, resolution: List[int]) -> "SceneManifest":
        if os.path.exists(os.path.join(project_dir, "manifest.json")):
            raise ManifestError(
                f"manifest.json already exists in {project_dir} — use load() "
                f"to resume an existing project, not create()."
            )
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "runs"), exist_ok=True)
        return cls(project_dir, resolution)

    @classmethod
    def load(cls, project_dir: str) -> "SceneManifest":
        path = os.path.join(project_dir, "manifest.json")
        if not os.path.exists(path):
            raise ManifestError(
                f"No manifest.json in {project_dir} — run `init` first."
            )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            project_dir,
            data["resolution"],
            revisions=data.get("revisions", []),
            objects=data.get("objects", {}),
        )

    def to_dict(self) -> dict:
        return {
            "resolution": self.resolution,
            "revisions": self.revisions,
            "objects": self.objects,
        }

    def save(self) -> None:
        """Atomic write: temp file + rename, so a crash mid-write never
        leaves manifest.json truncated or pointing at an unwritten image.
        Caller is responsible for writing the canvas/mask image(s) for a
        revision BEFORE calling save() — see pipelineInc.md §9."""
        os.makedirs(self.project_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=self.project_dir, prefix=".manifest_", suffix=".json.tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            os.replace(tmp_path, self.manifest_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    # ────────────────────────── revisions ──────────────────────────────

    def latest_revision(self) -> dict:
        if not self.revisions:
            raise ManifestError("No revisions yet — run `init` first.")
        return self.revisions[-1]

    def latest_revision_id(self) -> int:
        return self.latest_revision()["id"]

    def canvas_path(self, revision_id: int) -> str:
        return os.path.join(self.project_dir, f"canvas_v{revision_id}.png")

    def next_revision_id(self) -> int:
        return len(self.revisions)

    def add_revision(
        self,
        op: str,
        prompt: str,
        parent: Optional[int],
        object: Optional[str] = None,
        **extra,
    ) -> dict:
        rev_id = self.next_revision_id()
        rev = {
            "id": rev_id,
            "image": f"canvas_v{rev_id}.png",
            "op": op,
            "prompt": prompt,
            "parent": parent,
        }
        if object is not None:
            rev["object"] = object
        rev.update(extra)
        self.revisions.append(rev)
        return rev

    # ────────────────────────── objects ──────────────────────────────

    def mask_path(self, object_name: str) -> str:
        return os.path.join(self.project_dir, "masks", f"{object_name}.png")

    def get_object(self, name: str) -> Optional[dict]:
        return self.objects.get(name)

    def require_object(self, name: str) -> dict:
        obj = self.get_object(name)
        if obj is None:
            raise ManifestError(f"No object named '{name}' in this project's manifest.")
        if obj.get("retired_at") is not None:
            raise ManifestError(
                f"Object '{name}' was retired at revision {obj['retired_at']} "
                f"(op={self.revisions[obj['retired_at']].get('op')!r}) and can no "
                f"longer be targeted directly."
            )
        return obj

    def register_object(
        self,
        name: str,
        noun: str,
        created_at: int,
        parent_object: Optional[str] = None,
        replaces: Optional[str] = None,
    ) -> dict:
        if name in self.objects and self.objects[name].get("retired_at") is None:
            raise ManifestError(
                f"Object '{name}' already exists and is active — choose a "
                f"different name, or retire it first via replace/remove."
            )
        if parent_object is not None:
            self.require_object(parent_object)
        obj = {
            "noun": noun,
            "mask": os.path.relpath(self.mask_path(name), self.project_dir).replace(os.sep, "/"),
            "created_at": created_at,
            "retired_at": None,
            "parent_object": parent_object,
        }
        if replaces is not None:
            obj["replaces"] = replaces
        self.objects[name] = obj
        return obj

    def retire_object(self, name: str, retired_at: int) -> List[str]:
        """Retire an object (replace/remove). Returns names of any child
        objects that are now orphaned (parent_object cleared, flagged in
        the returned list rather than silently dropped) — see
        pipelineInc.md §1."""
        obj = self.require_object(name)
        obj["retired_at"] = retired_at
        orphaned = []
        for child_name, child in self.objects.items():
            if child.get("parent_object") == name and child.get("retired_at") is None:
                child["parent_object"] = None
                child["orphaned_from"] = name
                orphaned.append(child_name)
        return orphaned

    def active_objects(self) -> Dict[str, dict]:
        return {n: o for n, o in self.objects.items() if o.get("retired_at") is None}

    def children_of(self, name: str) -> List[str]:
        return [n for n, o in self.objects.items() if o.get("parent_object") == name]
