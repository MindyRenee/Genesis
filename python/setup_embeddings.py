#!/usr/bin/env python3
"""Download GloVe embeddings and filter to Genesis's vocabulary.

This is an OPTIONAL setup step. Genesis's embedding store works
without GloVe — it uses spectral graph embedding and TF-IDF from
her own concept network. Downloading GloVe adds distributional
semantics from general English, which improves pattern recognition
for words she hasn't explicitly learned.

Usage:
    python3 setup_embeddings.py [--data-dir DIR] [--dim 50]

If no data dir is specified, defaults to ~/.local/share/genesis.
"""

from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

# ─── Config ──────────────────────────────────────────────────────

GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DIM = 50  # 50d is compact and sufficient for semantic similarity


# ─── Main ────────────────────────────────────────────────────────


def _parse_setup_args() -> tuple[str, int]:
    """Parse --data-dir and --dim from command-line arguments."""
    data_dir = (
        sys.argv[sys.argv.index("--data-dir") + 1]
        if "--data-dir" in sys.argv
        else str(Path.home() / ".local" / "share" / "genesis")
    )
    dim = int(sys.argv[sys.argv.index("--dim") + 1]) if "--dim" in sys.argv else GLOVE_DIM
    return data_dir, dim


def _load_concept_vocab(data_dir: str) -> tuple[object, set[str]] | None:
    """Load the concept network and build vocabulary from concept names.

    Returns (network, vocab) or None if no saved state is found.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from genesis_cognitive.concepts import ConceptNetwork
    from genesis_cognitive.persistence import load_state, restore_network

    print(f"Loading concept network from {data_dir}...")
    data = load_state(data_dir)
    if not data:
        print("No saved state found! Run Genesis at least once first.")
        return None

    net = ConceptNetwork()
    restore_network(net, data["concept_network"])
    print(f"  {net.size} concepts, {net.edge_count} edges")

    # Build vocabulary from concept names
    vocab: set[str] = set()
    for concept_id in net._concepts:
        name = concept_id.replace("_", " ").replace("-", " ")
        for word in name.split():
            if len(word) > 1:
                vocab.add(word.lower())
    print(f"  Vocabulary: {len(vocab)} unique words")
    return net, vocab


def _download_and_extract_glove(tmpdir: str, dim: int) -> str:
    """Download GloVe zip and extract the target dimension file."""
    import urllib.request

    glove_filename = f"glove.6B.{dim}d.txt"
    zip_path = os.path.join(tmpdir, "glove.6B.zip")

    # Download with progress reporting
    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        """Print download progress at periodic intervals."""
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            if block_num % 1000 == 0:
                print(f"  {pct}% ({downloaded // (1024 * 1024)} MB)")

    urllib.request.urlretrieve(GLOVE_URL, zip_path, reporthook=reporthook)
    print("  Download complete.")

    # Extract only the file we need
    print(f"Extracting {glove_filename}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(glove_filename, tmpdir)

    return os.path.join(tmpdir, glove_filename)


def _filter_glove_vectors(
    glove_path: str, vocab: set[str]
) -> tuple[list[str], list[np.ndarray]] | None:
    """Parse GloVe file and filter to Genesis's vocabulary.

    Returns (words, vectors) or None if no words matched.
    """
    print(f"Filtering to Genesis's vocabulary ({len(vocab)} words)...")
    words: list[str] = []
    vectors: list[np.ndarray] = []
    found = 0

    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0].lower()
            if word not in vocab:
                continue
            vec = np.array(parts[1:], dtype=np.float32)
            words.append(word)
            vectors.append(vec)
            found += 1

            if found % 5000 == 0:
                print(f"  Found {found}/{len(vocab)} words...")

    print(f"  Matched: {found}/{len(vocab)} words")

    if found == 0:
        print("ERROR: No words matched.")
        return None

    return words, vectors


def _normalize_and_save_vectors(
    words: list[str], vectors: list[np.ndarray], output_path: str
) -> None:
    """L2-normalize vectors and save to npz file."""
    # Normalize all vectors (L2)
    print("Normalizing vectors...")
    matrix = np.stack(vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    matrix = matrix / norms

    # Save
    print(f"Saving to {output_path}...")
    np.savez(output_path, words=np.array(words), vectors=matrix)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Words: {len(words)}")
    print(f"  Dimensions: {matrix.shape[1]}")
    print("Done! Genesis will use these embeddings automatically next time she starts.")


def main() -> None:
    """Download GloVe embeddings and save them as an .npz file for Genesis."""
    data_dir, dim = _parse_setup_args()
    output_path = os.path.join(data_dir, "embeddings.npz")

    # Already exists?
    if os.path.exists(output_path):
        print(f"Embeddings already exist at {output_path}")
        print("Delete the file to re-download.")
        return

    result = _load_concept_vocab(data_dir)
    if result is None:
        return
    _net, vocab = result

    # Download GloVe
    print(f"\nDownloading GloVe ({dim}d) from {GLOVE_URL}...")
    print("This is a ~822 MB download. It may take several minutes.")

    with tempfile.TemporaryDirectory() as tmpdir:
        glove_path = _download_and_extract_glove(tmpdir, dim)
        filtered = _filter_glove_vectors(glove_path, vocab)
        if filtered is None:
            return
        words, vectors = filtered
        _normalize_and_save_vectors(words, vectors, output_path)


if __name__ == "__main__":
    main()
