"""Central configuration. All secrets come ONLY from environment variables.

This module is the single source of truth for runtime configuration. Nothing
else in the project should read ``os.environ`` directly for secrets — import
``settings`` from here instead.

Two hard guardrails are enforced at construction time:

1.  ``MODE`` is constrained to one of {"backtest", "paper", "live"} and
    defaults to "paper".
2.  ``MODE == "live"`` refuses to start unless ``LIVE_TRADING_CONFIRMED=true``
    is set explicitly in the environment. There is no in-code default that can
    flip a system into live trading.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Mode(str, Enum):
    """The three operating modes of the system.

    The order matters conceptually: backtest -> paper -> live is the build and
    promotion path described in the README.
    """

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    """Strongly-typed, environment-driven configuration.

    Reads from process environment and, for local development, a ``.env`` file
    (which must never be committed — see ``.gitignore``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Mode & safety gate -------------------------------------------------
    mode: Mode = Field(
        default=Mode.PAPER,
        description="Operating mode. Defaults to the safe 'paper' mode.",
    )
    live_trading_confirmed: bool = Field(
        default=False,
        description=(
            "Must be explicitly true in the environment for live mode to "
            "start. Acts as a deliberate human confirmation gate."
        ),
    )

    # --- Exchange credentials (secrets) -------------------------------------
    exchange_id: str = Field(default="binance")
    exchange_api_key: SecretStr = Field(default=SecretStr(""))
    exchange_api_secret: SecretStr = Field(default=SecretStr(""))
    exchange_api_password: SecretStr = Field(default=SecretStr(""))
    exchange_sandbox: bool = Field(default=True)

    # --- Database -----------------------------------------------------------
    database_url: str = Field(default="sqlite:///data/trading.db")

    # --- Features / Labels --------------------------------------------------
    # Forward horizon (in bars) for the label's forward log return.
    label_horizon: int = Field(
        default=1,
        gt=0,
        description="N: label the N-period-forward log return. Default 1 bar.",
    )
    # Full ROUND-TRIP cost the forward return must clear to be worth a Long.
    # 0.002 = 20bps taker, covering BOTH legs (buy + sell), not one.
    round_trip_cost: float = Field(
        default=0.002,
        ge=0,
        description="Round-trip (entry+exit) fee fraction. 0.002 = 20bps total.",
    )
    # Extra one-way slippage allowance, in basis points; counted on both legs
    # so the effective hurdle is round_trip_cost + 2 * (slippage_bps / 10_000).
    slippage_bps: float = Field(
        default=0.0,
        ge=0,
        description="Per-leg slippage in basis points; applied to both legs.",
    )
    # Minimum model confidence (P(Long)) for the trading daemon to act on a
    # long signal. Default 0.50 (the oos_signal default). NOT 0.755 — that was
    # a raw-score value from the failed turnover experiment, not a validated
    # confidence threshold. Raise it deliberately once a model shows edge.
    confidence_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="P(Long) the daemon must exceed before acting on a long.",
    )

    # --- Risk limits --------------------------------------------------------
    max_position_notional: float = Field(default=1000.0, gt=0)
    max_daily_loss: float = Field(default=200.0, gt=0)
    max_open_positions: int = Field(default=5, gt=0)
    max_leverage: float = Field(default=1.0, gt=0)

    # --- Risk gate / circuit breaker (HALT) ---------------------------------
    # Per-trade hard cap: a single order's notional may not exceed this
    # fraction of NAV. Matches the backtest's 20% fixed-fractional sizing.
    max_trade_fraction: float = Field(
        default=0.20, gt=0, le=1.0,
        description="Hard per-trade cap as a fraction of NAV (0.20 = 20%).",
    )
    # Total long exposure cap as a fraction of NAV. 0.20 = one full position
    # (spot, long-only, single symbol, no pyramiding) so the cap actually binds.
    max_total_exposure: float = Field(
        default=0.20, gt=0,
        description="Max summed position notional as a fraction of NAV.",
    )
    # HALT trips if rolling 24h drawdown is at or below this (negative) level.
    halt_drawdown_pct: float = Field(
        default=-3.0, le=0,
        description="Rolling-24h drawdown (%) at/below which SYSTEM_HALT trips.",
    )
    # HALT trips after this many consecutive ccxt/exchange failures.
    halt_max_consecutive_failures: int = Field(
        default=5, gt=0,
        description="Consecutive exchange failures that trip SYSTEM_HALT.",
    )
    # HALT trips if the heartbeat hasn't updated within this many minutes.
    halt_heartbeat_timeout_minutes: float = Field(
        default=15.0, gt=0,
        description="Minutes without a heartbeat update before HALT trips.",
    )

    # --- Web dashboard ------------------------------------------------------
    web_host: str = Field(default="127.0.0.1")
    web_port: int = Field(default=8000, gt=0, lt=65536)

    # --- Misc ---------------------------------------------------------------
    log_level: str = Field(default="INFO")
    base_currency: str = Field(default="USDT")

    # --- Validators ---------------------------------------------------------
    @model_validator(mode="after")
    def _enforce_live_gate(self) -> "Settings":
        """Refuse to construct live settings without explicit confirmation."""
        if self.mode is Mode.LIVE and not self.live_trading_confirmed:
            raise ValueError(
                "Refusing to start in LIVE mode: set environment variable "
                "LIVE_TRADING_CONFIRMED=true to confirm real-money trading. "
                "If you did not intend live trading, set MODE=paper."
            )
        return self

    @model_validator(mode="after")
    def _require_credentials_when_trading(self) -> "Settings":
        """Live mode needs real credentials; paper/backtest can run without."""
        if self.mode is Mode.LIVE:
            if not self.exchange_api_key.get_secret_value():
                raise ValueError(
                    "LIVE mode requires EXCHANGE_API_KEY to be set."
                )
            if not self.exchange_api_secret.get_secret_value():
                raise ValueError(
                    "LIVE mode requires EXCHANGE_API_SECRET to be set."
                )
        return self

    # --- Convenience properties --------------------------------------------
    @property
    def is_live(self) -> bool:
        return self.mode is Mode.LIVE

    @property
    def is_paper(self) -> bool:
        return self.mode is Mode.PAPER

    @property
    def is_backtest(self) -> bool:
        return self.mode is Mode.BACKTEST

    @property
    def label_hurdle(self) -> float:
        """Forward-return threshold a Long must clear to beat costs.

        The full round-trip fee plus per-leg slippage on BOTH legs:
        ``round_trip_cost + 2 * slippage_bps / 10_000``. A forward log return
        above this is labelled Long (1); otherwise Flat (0).
        """
        return self.round_trip_cost + 2.0 * self.slippage_bps / 10_000.0

    def exchange_credentials(self) -> dict[str, Optional[str]]:
        """ccxt-style credential dict, secrets unwrapped at the call site only."""
        return {
            "apiKey": self.exchange_api_key.get_secret_value() or None,
            "secret": self.exchange_api_secret.get_secret_value() or None,
            "password": self.exchange_api_password.get_secret_value() or None,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated Settings instance.

    Cached so the live-trading gate and credential checks run exactly once per
    process. Call ``get_settings.cache_clear()`` in tests that need to vary the
    environment.
    """
    return Settings()  # type: ignore[call-arg]
