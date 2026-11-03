#!/usr/bin/env bash
# Package the Genesis modules the ARC-AGI-3 agent imports (stdlib-only) and
# build the Kaggle notebook into kaggle/arc3/build/.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
mkdir -p "$here/build"
cd "$repo"
tar czf "$here/build/genesis_src.tar.gz" \
  genesis_client/__init__.py genesis_client/client.py genesis_client/exceptions.py \
  genesis_client/protocol.py genesis_client/swallow.py genesis_client/tensor.py \
  genesis_client/types.py \
  python/_views/__init__.py \
  python/basal_ganglia/__init__.py python/basal_ganglia/basal_ganglia.py \
  python/hypothalamus/__init__.py python/hypothalamus/cortisol.py python/hypothalamus/crh.py \
  python/hypothalamus/histamine.py python/hypothalamus/melatonin.py \
  python/hypothalamus/orexin.py python/hypothalamus/oxytocin.py \
  python/hypothalamus/transduction.py python/hypothalamus/vasopressin.py \
  python/parietal_lobe/__init__.py python/parietal_lobe/arc_ascii.py \
  python/parietal_lobe/arc_neuro.py python/parietal_lobe/arc_play.py \
  python/parietal_lobe/arc_policy.py python/parietal_lobe/arc_sources.py \
  python/thalamus/__init__.py python/thalamus/brain_waves.py
cp "$here/kernel-metadata.json" "$here/build/kernel-metadata.json"
python3 "$here/build_notebook.py"
echo "Push with: KAGGLE_API_TOKEN=... kaggle kernels push -p $here/build"
