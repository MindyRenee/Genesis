## Summary

<!-- What changed, and which invariant or behavior it protects. -->

## Root cause / rationale

<!-- Why this is the right fix, not just a symptom patch. -->

## Verification

- [ ] `cargo test`
- [ ] `ruff check`
- [ ] `python3 -m pyflakes python/ genesis_client/ infrastructure/ world/ eval/ *.py scripts/*.py`
- [ ] `python3 -m mypy python/ genesis_client/ infrastructure/ world/ eval/ --ignore-missing-imports`
- [ ] `PYTHONPATH=python:. python3 -m pytest python/tests/ -q -o addopts=''`
- [ ] Focused tests added or updated

## Evidence limits

<!-- State what was not tested or proven. For ARC work, distinguish local diagnostics from official scorecard evidence. -->
