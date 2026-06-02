"""Core trading primitives.

Public surface: the order gateway and its types. Strategy code should import
``submit_order`` and the ``Order`` types from here and nothing lower-level for
placing orders, so the risk chokepoint stays intact.
"""

from core.engine import (
    Order,
    OrderExecutor,
    OrderResult,
    OrderType,
    Side,
    submit_order,
)

__all__ = [
    "Order",
    "OrderExecutor",
    "OrderResult",
    "OrderType",
    "Side",
    "submit_order",
]
