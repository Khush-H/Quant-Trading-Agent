"""Tests for the configuration guardrails.

These assert the safety contract directly: mode defaulting, the live gate, and
credential requirements. They are the first line of defense against an
accidental live-trading misconfiguration.
"""

from __future__ import annotations

import pytest

from config.settings import Mode, Settings


def _make(**env) -> Settings:
    """Build Settings from an explicit dict, ignoring any ambient .env."""
    # _env_file=None prevents a developer's local .env from leaking into tests.
    return Settings(_env_file=None, **env)


def test_default_mode_is_paper():
    s = _make()
    assert s.mode is Mode.PAPER
    assert s.is_paper


def test_backtest_mode_ok_without_credentials():
    s = _make(mode="backtest")
    assert s.is_backtest


def test_live_requires_confirmation_flag():
    with pytest.raises(ValueError, match="LIVE_TRADING_CONFIRMED"):
        _make(
            mode="live",
            exchange_api_key="k",
            exchange_api_secret="s",
        )


def test_live_with_confirmation_but_no_keys_is_rejected():
    with pytest.raises(ValueError, match="EXCHANGE_API_KEY"):
        _make(mode="live", live_trading_confirmed=True)


def test_live_fully_configured_constructs():
    s = _make(
        mode="live",
        live_trading_confirmed=True,
        exchange_api_key="k",
        exchange_api_secret="s",
    )
    assert s.is_live


def test_confirmation_flag_alone_does_not_enable_live():
    # Setting the flag while staying in paper must remain paper — the flag is a
    # gate, not a switch.
    s = _make(mode="paper", live_trading_confirmed=True)
    assert s.is_paper


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        _make(mode="turbo")


def test_secrets_not_exposed_in_repr():
    s = _make(
        mode="paper",
        exchange_api_key="supersecret",
        exchange_api_secret="alsosecret",
    )
    assert "supersecret" not in repr(s)
    assert "alsosecret" not in repr(s)
