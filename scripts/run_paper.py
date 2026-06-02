"""Run the paper-trading loop (and/or the dashboard).

Forces MODE=paper. Paper trading uses live prices but simulated fills, and —
like every other mode — places orders only through
:func:`core.engine.submit_order`. Implemented in the paper phase.

Usage:
    python -m scripts.run_paper
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    os.environ["MODE"] = "paper"

    from config import Mode, get_settings

    settings = get_settings()
    if settings.mode is not Mode.PAPER:  # defensive
        sys.exit(f"Expected paper mode, got {settings.mode.value!r}.")

    print(f"[run_paper] mode={settings.mode.value} "
          f"exchange={settings.exchange_id} sandbox={settings.exchange_sandbox}")
    raise NotImplementedError("Implemented in the paper phase.")


if __name__ == "__main__":
    main()
