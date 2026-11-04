#!/usr/bin/env python3
"""Download GloVe embeddings and filter to Genesis's vocabulary.

This is an OPTIONAL setup step. Genesis's embedding store works
without GloVe — it uses spectral graph embedding and TF-IDF from
its own concept network. Downloading GloVe adds distributional
semantics from general English, which improves pattern recognition
for words it hasn't explicitly learned.

Usage:
    python3 setup_embeddings.py [--data-dir DIR] [--dim 50]

If no data dir is specified, it resolves to the canonical dir:
GENESIS_DATA_DIR if set, else ~/.local/share/genesis.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np

from infrastructure.config import default_data_dir

# ─── Config ──────────────────────────────────────────────────────

GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_DIM = 50  # 50d is compact and sufficient for semantic similarity


# ─── Main ────────────────────────────────────────────────────────


def _parse_setup_args() -> tuple[str, int]:
    """Parse --data-dir and --dim from command-line arguments.

    The default comes from ``default_data_dir()`` — building an
    embedding index against anything but the canonical concept set
    is silently wrong, not merely inconvenient.
    """
    data_dir = (
        sys.argv[sys.argv.index("--data-dir") + 1]
        if "--data-dir" in sys.argv
        else str(default_data_dir())
    )
    dim = int(sys.argv[sys.argv.index("--dim") + 1]) if "--dim" in sys.argv else GLOVE_DIM
    return data_dir, dim


def _load_concept_vocab(data_dir: str) -> tuple[object, set[str]] | None:
    """Load the concept network and build vocabulary from concept names.

    Returns (network, vocab) or None if no saved state is found.
    """
    # The brain-region packages live under ``python/``; the repository root
    # alone does not make ``temporal_lobe`` importable when this file is run
    # directly as ``python setup_embeddings.py``.
    python_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)

    from temporal_lobe.concepts import ConceptNetwork

    from infrastructure.persistence import load_state, restore_network

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
    """Download GloVe with retries and byte-range resume, then extract one dimension."""
    glove_filename = f"glove.6B.{dim}d.txt"
    zip_path = os.path.join(tmpdir, "glove.6B.zip")
    chunk_size = 1024 * 1024
    max_attempts = 5
    timeout_seconds = 60

    for attempt in range(1, max_attempts + 1):
        existing_size = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
        headers = {"User-Agent": "Genesis-embedding-setup/1.0"}
        if existing_size:
            headers["Range"] = f"bytes={existing_size}-"
        request = urllib.request.Request(GLOVE_URL, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = getattr(response, "status", response.getcode())
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    range_parts = content_range.removeprefix("bytes ").split("/")
                    byte_range = range_parts[0].split("-") if range_parts else []
                    if len(byte_range) != 2 or int(byte_range[0]) != existing_size:
                        raise OSError(
                            "Download server returned an invalid byte range; "
                            "refusing to append potentially corrupt data."
                        )
                    total_size = (
                        int(range_parts[1])
                        if len(range_parts) == 2 and range_parts[1].isdigit()
                        else None
                    )
                    mode = "ab"
                elif status == 200:
                    # The server ignored Range; restart rather than append a full response.
                    existing_size = 0
                    total_header = response.headers.get("Content-Length")
                    total_size = (
                        int(total_header)
                        if total_header and total_header.isdigit()
                        else None
                    )
                    mode = "wb"
                else:
                    raise OSError(f"Unexpected HTTP status {status} while downloading GloVe.")

                downloaded = existing_size
                last_reported_mb = downloaded // (50 * 1024 * 1024)
                with open(zip_path, mode) as output:
                    while chunk := response.read(chunk_size):
                        output.write(chunk)
                        downloaded += len(chunk)
                        current_mb = downloaded // (50 * 1024 * 1024)
                        if current_mb > last_reported_mb:
                            if total_size:
                                pct = min(100, downloaded * 100 // total_size)
                                print(f"  {pct}% ({downloaded // (1024 * 1024)} MB)")
                            else:
                                print(f"  Downloaded {downloaded // (1024 * 1024)} MB")
                            last_reported_mb = current_mb

                if total_size is not None and downloaded != total_size:
                    raise OSError(
                        f"Incomplete GloVe archive: received {downloaded} of {total_size} bytes."
                    )
                break

        except urllib.error.HTTPError as exc:
            # A completed partial transfer can be followed by a 416 on resume.
            if exc.code == 416 and zipfile.is_zipfile(zip_path):
                break
            if exc.code == 416 and os.path.exists(zip_path):
                os.remove(zip_path)
            if attempt == max_attempts:
                raise RuntimeError(
                    f"GloVe download failed after {max_attempts} attempts: HTTP {exc.code}"
                ) from exc
            print(
                f"  Download attempt {attempt}/{max_attempts} failed "
                f"(HTTP {exc.code}); retrying..."
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"GloVe download failed after {max_attempts} attempts: {exc}"
                ) from exc
            print(
                f"  Download attempt {attempt}/{max_attempts} interrupted; "
                "retaining partial data and retrying..."
            )

        if attempt < max_attempts:
            time.sleep(min(2 ** (attempt - 1), 16))

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError("Downloaded GloVe file is not a valid ZIP archive.")

    print("  Download complete.")
    print(f"Extracting {glove_filename}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        if glove_filename not in zf.namelist():
            raise RuntimeError(
                f"{glove_filename} is not present in the downloaded GloVe archive."
            )
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
    print("Done! Genesis will use these embeddings automatically next time it starts.")


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

    # Keep the archive between runs so an interrupted mobile download can resume.
    # Remove this cache only after the filtered embeddings have been saved.
    download_dir = os.path.join(data_dir, ".glove-download")
    os.makedirs(download_dir, exist_ok=True)

    glove_path = _download_and_extract_glove(download_dir, dim)
    filtered = _filter_glove_vectors(glove_path, vocab)
    if filtered is None:
        return
    words, vectors = filtered
    _normalize_and_save_vectors(words, vectors, output_path)
    shutil.rmtree(download_dir)


if __name__ == "__main__":
    main()
