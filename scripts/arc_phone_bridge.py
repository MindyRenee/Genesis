"""Start the local Genesis ARC phone bridge."""
from __future__ import annotations

import argparse

from parietal_lobe.arc_phone_bridge import serve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default="127.0.0.1", help="loopback address (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--mind-socket",
        help="connect a headless Genesis Mind to an existing daemon socket",
    )
    args = parser.parse_args()
    serve(args.host, args.port, mind_socket=args.mind_socket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
