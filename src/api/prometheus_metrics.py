"""Convert Amgix cluster Metrics snapshots to Prometheus / OpenMetrics text exposition."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Iterable

from src.core.common.metrics_definitions import METRIC_DEFINITIONS, MetricKey
from src.core.models.cluster import Metrics

_PROMETHEUS_WINDOW_SECONDS = 60


def _fmt_sample_float(v: float) -> str:
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    # Compact decimal without trailing zeros noise where possible
    s = f"{v:.12g}"
    if s == "-0":
        return "0"
    return s


def _escape_help_line(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", " ").replace("\r", " ").strip()


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")


def _prometheus_metric_name(key: str) -> str:
    if not key:
        return "amgix_unknown"
    parts: list[str] = []
    for ch in key:
        if ch.isascii() and (ch.isalnum() or ch == "_"):
            parts.append(ch)
        else:
            parts.append("_")
    body = "".join(parts).strip("_") or "unknown"
    if body[0].isdigit():
        body = f"_{body}"
    return f"amgix_{body}"


def _dim_labels(dims: list[str]) -> list[tuple[str, str]]:
    if len(dims) == 3:
        return [
            ("vector_type", dims[0]),
            ("model", dims[1]),
            ("revision", dims[2]),
        ]
    return [(f"dim_{i}", v) for i, v in enumerate(dims)]


def _format_label_pairs(pairs: Iterable[tuple[str, str]]) -> str:
    ordered = sorted(pairs, key=lambda kv: kv[0])
    parts = [f'{k}="{_escape_label_value(v)}"' for k, v in ordered]
    return "{" + ",".join(parts) + "}" if parts else ""


def metrics_to_prometheus_text(metrics: Metrics, *, window_seconds: int = _PROMETHEUS_WINDOW_SECONDS) -> str:
    """Serialize a Metrics snapshot to Prometheus text exposition (gauges, fixed window)."""
    if window_seconds not in (30, 60):
        raise ValueError("window_seconds must be 30 or 60")

    lines: list[str] = []
    help_names: OrderedDict[str, str] = OrderedDict()

    def note_metric(name: str, help_text: str) -> None:
        help_names.setdefault(name, help_text)

    window_gauge = "amgix_metrics_window_seconds"
    note_metric(
        window_gauge,
        "Length in seconds of the rolling window used for live Amgix samples in this Prometheus exposition.",
    )

    # Pre-register HELP/TYPE for value and optional _n series
    for node_hostname, node in metrics.nodes.items():
        for series in node.metrics:
            base = _prometheus_metric_name(series.key)
            try:
                mk = MetricKey(series.key)
                desc = METRIC_DEFINITIONS[mk].description
            except (ValueError, KeyError):
                desc = f"Amgix metric {series.key}."
            note_metric(base, desc)
            ws = series.windows.get(window_seconds)
            if ws is not None and ws.n is not None:
                note_metric(f"{base}_n", f"Sample count (denominator) for {base}.")

    for name, help_text in help_names.items():
        lines.append(f"# HELP {name} {_escape_help_line(help_text)}")
        lines.append(f"# TYPE {name} gauge")

    lines.append(f"{window_gauge} {_fmt_sample_float(float(window_seconds))}")

    for node_hostname, node in metrics.nodes.items():
        for series in node.metrics:
            ws = series.windows.get(window_seconds)
            if ws is None:
                continue
            base = _prometheus_metric_name(series.key)
            label_pairs: list[tuple[str, str]] = [
                ("host", node_hostname),
                ("role", node.role),
                ("leader", "true" if node.is_leader else "false"),
            ]
            label_pairs.extend(_dim_labels(list(series.dims)))
            lbl = _format_label_pairs(label_pairs)
            lines.append(f"{base}{lbl} {_fmt_sample_float(float(ws.value))}")
            if ws.n is not None:
                lines.append(f"{base}_n{lbl} {_fmt_sample_float(float(ws.n))}")

    lines.append("")
    return "\n".join(lines)
