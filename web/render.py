"""Plain-Python HTML rendering for the dashboard (no template engine).

Keeps the dashboard's runtime dependencies to FastAPI/uvicorn: there is no
Jinja2. The page shell lives in ``templates/index.html`` with ``{{ NAME }}``
slots; :func:`render_index` fills them with fragments built here.

EVERY dynamic value is HTML-escaped via :func:`html.escape` before it reaches
the page. That matters most for free-text fields that originate in the database
— the execution ``reason`` and the halt ``reason`` — which must never be able to
inject markup. Numbers are formatted, not trusted as-is.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Optional

_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"


def _esc(value: Any) -> str:
    """HTML-escape any value (None -> empty)."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _fmt_pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:+.2f}%"


def _sign_class(value: Optional[float]) -> str:
    if value is None:
        return "muted"
    if value > 0:
        return "pos"
    if value < 0:
        return "neg"
    return ""


def _fmt_num(value: Any, spec: str, *, dash_if_none: bool = False) -> str:
    if value is None:
        return "—" if dash_if_none else format(0.0, spec)
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return _esc(value)


def _live_warning(is_live: bool) -> str:
    if not is_live:
        return ""
    return (
        '<p class="live-warning">'
        "⚠ LIVE mode — real funds are at risk. Orders still route through the "
        "risk layer, but verify limits before trading."
        "</p>"
    )


_RESET_MESSAGES = {
    "cleared": "SYSTEM_HALT cleared. The daemon may take new entries on its "
               "next cycle.",
    "not_halted": "SYSTEM_HALT was not set — nothing to clear.",
    "missing_confirm": "Reset not performed: confirmation was required.",
}


def _reset_notice(notice: Optional[str]) -> str:
    msg = _RESET_MESSAGES.get(notice or "")
    if not msg:
        return ""
    return f'<p class="notice">{_esc(msg)}</p>'


def _halt_panel(halted: bool, halt_reason: Optional[str]) -> str:
    if halted:
        reason_html = (
            f'<div class="reason">Reason: {_esc(halt_reason)}</div>'
            if halt_reason else ""
        )
        # The ONLY state-changing control. The button POSTs an explicit confirm;
        # the server still enforces the guard and routes through clear_halt.
        button = (
            '<form method="post" action="/reset-halt" '
            "onsubmit=\"return confirm('Clear SYSTEM_HALT? Only do this after "
            "investigating the cause.');\">"
            '<input type="hidden" name="confirm" value="true" />'
            '<button type="submit">Reset HALT</button>'
            "</form>"
        )
        return (
            '<div class="halt tripped">'
            f'<div><span class="state">⛔ SYSTEM HALT — TRIPPED</span>{reason_html}</div>'
            f"{button}"
            "</div>"
        )
    return (
        '<div class="halt ok">'
        '<div><span class="state">✓ SYSTEM OK</span></div>'
        "</div>"
    )


def _positions_table(positions: list[dict]) -> str:
    if not positions:
        return '<p class="placeholder">Flat — no open positions.</p>'
    rows = []
    for p in positions:
        rows.append(
            "<tr>"
            f'<td class="l">{_esc(p.get("symbol"))}</td>'
            f'<td>{_fmt_num(p.get("quantity"), ".8f")}</td>'
            f'<td>{_fmt_num(p.get("avg_entry_price"), ".2f")}</td>'
            f'<td class="{_sign_class(p.get("realized_pnl"))}">'
            f'{_fmt_num(p.get("realized_pnl"), ".2f")}</td>'
            f'<td class="l muted">{_esc(p.get("updated_at"))}</td>'
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        '<th class="l">Symbol</th><th>Quantity</th><th>Avg Entry</th>'
        '<th>Realized PnL</th><th class="l">Updated</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _executions_table(executions: list[dict]) -> str:
    if not executions:
        return '<p class="placeholder">No executions logged yet.</p>'
    rows = []
    for e in executions:
        accepted = bool(e.get("accepted"))
        action = str(e.get("action") or "")
        # Constrain the tag class to the known set so a bad value can't smuggle
        # a class name; the text itself is escaped regardless.
        tag = action if action in {"buy", "sell", "hold"} else "hold"
        rows.append(
            f'<tr class="{"" if accepted else "rejected"}">'
            f'<td class="l muted">{_esc(e.get("decided_at"))}</td>'
            f'<td class="l">{_esc(e.get("symbol"))}</td>'
            f'<td class="l"><span class="tag {tag}">{_esc(action)}</span></td>'
            f'<td>{_fmt_num(e.get("price"), ".2f", dash_if_none=True)}</td>'
            f'<td>{_fmt_num(e.get("quantity"), ".8f")}</td>'
            f'<td>{_fmt_num(e.get("notional"), ".2f")}</td>'
            f'<td>{_fmt_num(e.get("fee"), ".4f")}</td>'
            f'<td>{_fmt_num(e.get("slippage"), ".4f")}</td>'
            f'<td class="l">{"accepted" if accepted else "rejected"}</td>'
            # Free text straight from the DB — escaped so it can never inject markup.
            f'<td class="l muted">{_esc(e.get("reason"))}</td>'
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        '<th class="l">Decided (UTC)</th><th class="l">Symbol</th>'
        '<th class="l">Action</th><th>Price</th><th>Qty</th><th>Notional</th>'
        '<th>Fee</th><th>Slippage</th><th class="l">Status</th>'
        '<th class="l">Reason</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def render_index(snapshot: dict, *, reset_notice: Optional[str] = None) -> str:
    """Render the dashboard page from a read-only snapshot.

    ``snapshot`` is the dict from :meth:`web.data.DashboardData.snapshot`. All
    dynamic values are escaped/formatted here; nothing is interpolated raw.
    """
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")

    halted = bool(snapshot.get("halted"))
    mode = str(snapshot.get("mode") or "")
    # Whitelist the mode token used in CSS class / banner so it can't break out.
    mode_token = mode if mode in {"paper", "backtest", "live"} else "paper"
    pnl = snapshot.get("pnl_24h_pct")
    dd = snapshot.get("drawdown_24h_pct")

    fields = {
        "BODY_CLASS": "halted" if halted else "",
        "MODE": _esc(mode_token),
        "RESET_NOTICE": _reset_notice(reset_notice),
        "LIVE_WARNING": _live_warning(bool(snapshot.get("is_live"))),
        "HALT_PANEL": _halt_panel(halted, snapshot.get("halt_reason")),
        "PNL_24H": _fmt_pct(pnl),
        "PNL_CLASS": _sign_class(pnl),
        "DRAWDOWN_24H": _fmt_pct(dd),
        "DD_CLASS": _sign_class(dd),
        "HEARTBEAT": _esc(snapshot.get("last_heartbeat_iso") or "—"),
        "POSITIONS": _positions_table(snapshot.get("positions") or []),
        "EXECUTIONS": _executions_table(snapshot.get("executions") or []),
        "HALTED_JS": "true" if halted else "false",
    }

    for name, value in fields.items():
        template = template.replace("{{ " + name + " }}", value)
    return template
