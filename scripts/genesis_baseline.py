"""Genesis baseline + drift + chemical-assay CLI.

Captures resting baselines, checks a live build against a baseline,
and maps what each of the 18 neurochemicals actually does — always
against an isolated throwaway daemon, never the production instance.

Usage:
    python3 scripts/genesis_baseline.py capture --out baseline.json
    python3 scripts/genesis_baseline.py check --baseline baseline.json
    python3 scripts/genesis_baseline.py assay --out assay.json
    python3 scripts/genesis_baseline.py sanity
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)


def _cmd_capture(args: argparse.Namespace) -> int:
    from genesis_cognitive.eval import capture_snapshot, isolated_daemon

    with isolated_daemon() as client:
        snap = capture_snapshot(client, settle_ticks=args.settle, settle_dt=args.dt)
    snap.save(args.out)
    print(f"captured baseline -> {args.out}")
    print(f"  phase={snap.phase_name} arousal={snap.arousal:.3f} "
          f"valence={snap.valence:.3f} plasticity={snap.plasticity_gate:.3f}")
    print(f"  coupling_drift={snap.coupling_drift:.4f}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    from genesis_cognitive.eval import (
        Snapshot,
        capture_snapshot,
        compare_snapshots,
        isolated_daemon,
    )

    baseline = Snapshot.load(args.baseline)
    with isolated_daemon() as client:
        current = capture_snapshot(client, settle_ticks=args.settle, settle_dt=args.dt)
    report = compare_snapshots(baseline, current)
    for line in report.summary_lines():
        print(line)
    if args.show_ok:
        for e in report.entries:
            if e.severity == "ok":
                print(f"  [OK  ] {e.field}: {e.current}")
    return 0 if report.passed else 1


def _cmd_sanity(args: argparse.Namespace) -> int:
    from genesis_cognitive.eval import (
        capture_snapshot,
        check_snapshot_vs_defaults,
        isolated_daemon,
    )

    with isolated_daemon() as client:
        snap = capture_snapshot(client, settle_ticks=args.settle, settle_dt=args.dt)
    report = check_snapshot_vs_defaults(snap)
    for line in report.summary_lines():
        print(line)
    return 0 if report.passed else 1


def _cmd_assay(args: argparse.Namespace) -> int:
    from genesis_cognitive.eval import AssayResult, isolated_daemon, run_chemical_assay

    with isolated_daemon() as client:
        result = run_chemical_assay(
            client,
            impulse=args.impulse,
            settle_ticks=args.settle,
            washout_ticks=args.washout,
            dt=args.dt,
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote assay matrix -> {args.out}")
    else:
        result = AssayResult.from_dict(result.to_dict())  # round-trip sanity
    for line in result.summary_lines():
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Genesis baseline / drift / chemical-assay harness "
        "(isolated daemon only)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cap = sub.add_parser("capture", help="capture a resting baseline JSON file")
    p_cap.add_argument("--out", required=True, help="output baseline JSON path")
    p_cap.add_argument("--settle", type=int, default=60, help="settle ticks (default 60)")
    p_cap.add_argument("--dt", type=float, default=1.0, help="seconds per tick (default 1.0)")
    p_cap.set_defaults(func=_cmd_capture)

    p_check = sub.add_parser("check", help="compare a fresh capture against a baseline")
    p_check.add_argument("--baseline", required=True, help="baseline JSON path")
    p_check.add_argument("--settle", type=int, default=60, help="settle ticks (default 60)")
    p_check.add_argument("--dt", type=float, default=1.0, help="seconds per tick (default 1.0)")
    p_check.add_argument("--show-ok", action="store_true", help="also print passing fields")
    p_check.set_defaults(func=_cmd_check)

    p_sanity = sub.add_parser(
        "sanity", help="check resting state against genetic defaults (no baseline file)"
    )
    p_sanity.add_argument("--settle", type=int, default=60, help="settle ticks (default 60)")
    p_sanity.add_argument("--dt", type=float, default=1.0, help="seconds per tick (default 1.0)")
    p_sanity.set_defaults(func=_cmd_sanity)

    p_assay = sub.add_parser("assay", help="run the 18-chemical impulse-response assay")
    p_assay.add_argument("--out", default=None, help="output assay JSON path (optional)")
    p_assay.add_argument("--impulse", type=float, default=0.30, help="impulse dose (default 0.30)")
    p_assay.add_argument("--settle", type=int, default=60, help="initial settle ticks (default 60)")
    p_assay.add_argument("--washout", type=int, default=20, help="washout ticks (default 20)")
    p_assay.add_argument("--dt", type=float, default=1.0, help="seconds per tick (default 1.0)")
    p_assay.set_defaults(func=_cmd_assay)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
