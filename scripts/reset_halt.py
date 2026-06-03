"""Manually clear SYSTEM_HALT — the ONLY way the circuit breaker resets.

SYSTEM_HALT never self-clears in the engine or risk path. Once it trips (24h
drawdown, consecutive exchange failures, or a stale heartbeat), the daemon
flattens to cash and refuses all new entries until an operator runs this script
after investigating the cause.

Usage:
    python -m scripts.reset_halt              # show status
    python -m scripts.reset_halt --confirm    # clear the halt
"""

from __future__ import annotations

import argparse

from config import get_settings
from core.database import Database


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset the SYSTEM_HALT flag.")
    parser.add_argument(
        "--confirm", action="store_true",
        help="Actually clear the halt. Without this, only the status is shown.",
    )
    args = parser.parse_args()

    settings = get_settings()
    db = Database(settings)
    db.init_schema()

    if not db.is_halted():
        print("[reset_halt] SYSTEM_HALT is not set. Nothing to do.")
        return

    reason = db.halt_reason() or "(no reason recorded)"
    print(f"[reset_halt] SYSTEM_HALT is SET. Reason: {reason}")
    if not args.confirm:
        print("[reset_halt] Re-run with --confirm to clear it once the cause is "
              "resolved. (Resetting does not fix the underlying condition.)")
        return

    db.clear_halt()
    db.reset_exchange_failures()
    print("[reset_halt] SYSTEM_HALT cleared. The daemon may take new entries "
          "again on its next cycle.")


if __name__ == "__main__":
    main()
