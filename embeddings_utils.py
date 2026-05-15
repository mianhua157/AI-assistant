from __future__ import annotations

import os
from pathlib import Path


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _candidate_cache_roots() -> list[Path]:
    roots: list[Path] = []

    for env_name in ("EMBEDDING_MODEL_PATH", "SENTENCE_TRANSFORMERS_HOME", "HF_HOME", "TRANSFORMERS_CACHE"):
        value = os.getenv(env_name)
        if value:
            roots.append(Path(value))

    home = Path.home()
    roots.extend(
        [
            home / ".cache" / "huggingface",
            home / ".cache" / "torch" / "sentence_transformers",
        ]
    )

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            seen.add(root)
            deduped.append(root)
    return deduped


def resolve_local_embedding_model_path(model_name: str) -> Path | None:
    explicit_path = os.getenv("EMBEDDING_MODEL_PATH")
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.exists():
            return candidate

    model_tail = model_name.split("/")[-1]
    repo_dir_name = model_name.replace("/", "--")

    for root in _candidate_cache_roots():
        direct_candidate = root / model_tail
        if (direct_candidate / "modules.json").exists():
            return direct_candidate

        hub_root = root if root.name == "hub" else root / "hub"
        snapshot_root = hub_root / f"models--{repo_dir_name}" / "snapshots"
        if not snapshot_root.exists():
            continue

        snapshots = sorted(
            [path for path in snapshot_root.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            if (snapshot / "modules.json").exists():
                return snapshot

    return None


def load_sentence_transformer(model_name: str = DEFAULT_EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    local_path = resolve_local_embedding_model_path(model_name)
    if local_path is not None:
        print(f"Loading embedding model from local path: {local_path}")
        return SentenceTransformer(str(local_path), local_files_only=True)

    try:
        print("Loading embedding model from local cache if available...")
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the embedding model from local files. "
            "Set EMBEDDING_MODEL_PATH to a local SentenceTransformer directory, "
            "or pre-download the model before running offline."
        ) from exc
