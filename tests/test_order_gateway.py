"""Tests for the single order gateway.

Locks in the contract that risk runs before any executor and that mode/executor
mismatches are rejected. As real risk rules land, extend these — but never
delete the "risk runs first" assertions.
"""

from __future__ import annotations

from config.settings import Mode, Settings
from core.engine import Order, OrderResult, Side, submit_order
from core.risk import RiskDecision


def _settings(mode: str = "paper") -> Settings:
    return Settings(_env_file=None, mode=mode)


class _RecordingExecutor:
    """Executor stub that records whether it was called."""

    def __init__(self, mode: Mode) -> None:
        self.mode = mode
        self.called_with: Order | None = None

    def execute(self, order: Order) -> OrderResult:
        self.called_with = order
        return OrderResult(
            order=order, accepted=True, mode=self.mode, filled_quantity=order.quantity
        )


class _ApproveRisk:
    def check(self, order, *, settings=None) -> RiskDecision:
        return RiskDecision(approved=True, order=order)


class _RejectRisk:
    def check(self, order, *, settings=None) -> RiskDecision:
        return RiskDecision(approved=False, order=order, reason="nope")


class _ResizeRisk:
    """Approves but halves the quantity, to prove resizing is honored."""

    def check(self, order, *, settings=None) -> RiskDecision:
        return RiskDecision(approved=True, order=order.with_quantity(order.quantity / 2))


def _order(qty: float = 1.0) -> Order:
    return Order(symbol="BTC/USDT", side=Side.BUY, quantity=qty)


def test_rejected_order_never_reaches_executor():
    settings = _settings("paper")
    ex = _RecordingExecutor(Mode.PAPER)
    result = submit_order(_order(), ex, risk=_RejectRisk(), settings=settings)
    assert result.accepted is False
    assert result.reason == "nope"
    assert ex.called_with is None  # the executor must not have run


def test_approved_order_reaches_executor():
    settings = _settings("paper")
    ex = _RecordingExecutor(Mode.PAPER)
    result = submit_order(_order(2.0), ex, risk=_ApproveRisk(), settings=settings)
    assert result.accepted is True
    assert ex.called_with is not None
    assert ex.called_with.quantity == 2.0


def test_risk_resize_is_forwarded_to_executor():
    settings = _settings("paper")
    ex = _RecordingExecutor(Mode.PAPER)
    submit_order(_order(4.0), ex, risk=_ResizeRisk(), settings=settings)
    assert ex.called_with is not None
    assert ex.called_with.quantity == 2.0  # halved by risk


def test_executor_mode_must_match_configured_mode():
    settings = _settings("paper")
    wrong = _RecordingExecutor(Mode.LIVE)  # live executor under paper config
    result = submit_order(_order(), wrong, risk=_ApproveRisk(), settings=settings)
    assert result.accepted is False
    assert "does not match" in (result.reason or "")
    assert wrong.called_with is None
