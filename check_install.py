"""Report module versions and the backends that will be used.

Run after installation or upgrade to confirm the checkout is consistent.
"""
from __future__ import annotations

from pathlib import Path

from backtester.optimize import active_backend
from backtester.pipeline.store import STORE_VERSION, _duck_available, open_store

EXPECTED_STORE_VERSION = "2026.08.3"


def main() -> None:
    print(f"store module version : {STORE_VERSION}", end="")
    if STORE_VERSION != EXPECTED_STORE_VERSION:
        print(f"   MISMATCH (expected {EXPECTED_STORE_VERSION})")
    else:
        print("   OK")
    print(f"duckdb available     : {_duck_available()}")
    print(f"optimiser backend    : {active_backend()}")

    for p in ("store",):
        root = Path(p)
        if not root.exists():
            print(f"store '{p}'          : not built yet "
                  f"(run: python3 build_store.py --source fixture)")
            continue
        s = open_store(p)
        print(f"store '{p}'          : {type(s).__name__}, has_bars={s.has_bars()}, "
              f"symbols={len(s.symbols())}")


if __name__ == "__main__":
    main()
