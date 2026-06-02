"""Run LIVE trading. Real money. Read this before touching it.

Live mode is gated three ways, and this script does NOT relax any of them — it
verifies them loudly:

  1. MODE must be "live" (set in the environment, not defaulted here). This
     script deliberately does NOT set MODE for you; you must opt in.
  2. LIVE_TRADING_CONFIRMED must be exactly "true" (enforced in
     config.settings — Settings construction raises otherwise).
  3. Exchange credentials must be present (also enforced in config.settings).

Every order still flows through core.engine.submit_order, so the risk layer
applies in live exactly as in backtest and paper. Implemented in the live phase,
which is the LAST step of the build.

Usage (must set the env vars yourself, e.g. via a non-committed .env):
    MODE=live LIVE_TRADING_CONFIRMED=true python -m scripts.run_live
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    # Intentionally do NOT set MODE here. Refuse to proceed unless the operator
    # has explicitly chosen live mode in the environment.
    if os.environ.get("MODE") != "live":
        sys.exit(
            "Refusing to run: MODE is not 'live'. Set MODE=live explicitly to "
            "trade real funds. (Paper trading: use scripts/run_paper.py.)"
        )
    if os.environ.get("LIVE_TRADING_CONFIRMED") != "true":
        sys.exit(
            "Refusing to run: LIVE_TRADING_CONFIRMED is not 'true'. This is the "
            "deliberate confirmation gate for real-money trading."
        )

    # Settings construction re-validates both gates and the credentials; if any
    # check fails this raises before any exchange connection is made.
    from config import Mode, get_settings

    settings = get_settings()
    if settings.mode is not Mode.LIVE:  # belt-and-suspenders
        sys.exit(f"Expected live mode, got {settings.mode.value!r}.")

    print("[run_live] LIVE MODE CONFIRMED — real funds at risk.")
    print(f"[run_live] exchange={settings.exchange_id} "
          f"sandbox={settings.exchange_sandbox}")
    raise NotImplementedError("Implemented in the live phase (final step).")


if __name__ == "__main__":
    main()
