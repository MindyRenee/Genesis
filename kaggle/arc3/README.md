# Genesis on Kaggle: ARC Prize 2026, ARC-AGI-3

`my_agent.py` is the `MyAgent` class the Kaggle gateway runs (the
[ARC-AGI-3 Kaggle Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
contract). Every decision comes from Genesis' own
`parietal_lobe.arc_policy.ARC3ExplorationPolicy`; the adapter only converts
frames and actions. No internet, no language model, no per-game code.

Build and push (Python 3.12, `kaggle>=2.2`):

```bash
kaggle/arc3/bundle.sh                       # -> kaggle/arc3/build/
KAGGLE_API_TOKEN=KGAT_... kaggle kernels push -p kaggle/arc3/build
KAGGLE_API_TOKEN=KGAT_... kaggle competitions submit arc-prize-2026-arc-agi-3 \
    -k mindybeatty/genesis-arc-agi-3 -v <version> -f submission.parquet -m "..."
```

Local run: copy `my_agent.py` into the starter's `agent/`, unpack
`build/genesis_src.tar.gz` to `/tmp/genesis_src`, then `make play-local`.
Knobs: `GENESIS_MAX_ACTIONS` (default 2500 per game), `GENESIS_MAX_SECONDS`
(default 1800 per game).
