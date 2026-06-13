"""Anomaly detection using statistical methods (z-score, threshold, IQR)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricPoint:
    """A single metric observation."""

    name: str
    value: float
    timestamp: float = 0.0  # Unix epoch seconds
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
        }


@dataclass
class Anomaly:
    """A detected anomaly."""

    metric_name: str
    value: float
    method: str  # zscore | threshold | iqr
    score: float  # how anomalous (z-score value, or deviation ratio)
    threshold: float  # the threshold that was breached
    message: str
    timestamp: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "method": self.method,
            "score": self.score,
            "threshold": self.threshold,
            "message": self.message,
            "timestamp": self.timestamp,
            "labels": self.labels,
        }


def mean(values: list[float]) -> float:
    """Calculate arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def stdev(values: list[float], mu: float | None = None) -> float:
    """Calculate population standard deviation."""
    if len(values) < 2:
        return 0.0
    if mu is None:
        mu = mean(values)
    variance = sum((x - mu) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def z_score(value: float, mu: float, sigma: float) -> float:
    """Calculate z-score for a single value."""
    if sigma == 0:
        return 0.0
    return (value - mu) / sigma


def iqr_bounds(values: list[float], multiplier: float = 1.5) -> tuple[float, float]:
    """Calculate IQR-based lower and upper bounds."""
    if len(values) < 4:
        return (float("-inf"), float("inf"))
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[3 * n // 4]
    iqr = q3 - q1
    return (q1 - multiplier * iqr, q3 + multiplier * iqr)


class AnomalyDetector:
    """Configurable anomaly detector supporting z-score, threshold, and IQR methods."""

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        window_size: int = 50,
        threshold_rules: dict[str, float] | None = None,
        iqr_multiplier: float = 1.5,
    ):
        self.zscore_threshold = zscore_threshold
        self.window_size = window_size
        self.threshold_rules = threshold_rules or {}
        self.iqr_multiplier = iqr_multiplier
        self._windows: dict[str, list[float]] = {}

    def _get_window(self, name: str) -> list[float]:
        return self._windows.get(name, [])

    def _update_window(self, name: str, value: float) -> list[float]:
        """Add value to the rolling window and return the updated window."""
        if name not in self._windows:
            self._windows[name] = []
        window = self._windows[name]
        window.append(value)
        if len(window) > self.window_size:
            self._windows[name] = window[-self.window_size:]
        return self._windows[name]

    def _check_zscore(self, point: MetricPoint, window: list[float]) -> Anomaly | None:
        """Check for z-score anomaly using the provided window."""
        if len(window) < 3:
            return None
        mu = mean(window)
        sigma = stdev(window, mu)
        z = z_score(point.value, mu, sigma)
        if abs(z) >= self.zscore_threshold:
            return Anomaly(
                metric_name=point.name,
                value=point.value,
                method="zscore",
                score=round(z, 3),
                threshold=self.zscore_threshold,
                message=f"Z-score {z:.2f} exceeds threshold {self.zscore_threshold} "
                f"(mean={mu:.2f}, stdev={sigma:.2f})",
                timestamp=point.timestamp,
                labels=point.labels,
            )
        return None

    def _check_threshold(self, point: MetricPoint) -> Anomaly | None:
        """Check for static threshold anomaly."""
        threshold = self.threshold_rules.get(point.name)
        if threshold is None:
            return None
        if point.value > threshold:
            return Anomaly(
                metric_name=point.name,
                value=point.value,
                method="threshold",
                score=point.value / threshold if threshold else 0.0,
                threshold=threshold,
                message=f"Value {point.value} exceeds threshold {threshold}",
                timestamp=point.timestamp,
                labels=point.labels,
            )
        return None

    def _check_iqr(self, point: MetricPoint, window: list[float]) -> Anomaly | None:
        """Check for IQR anomaly using the provided window."""
        if len(window) < 4:
            return None
        lower, upper = iqr_bounds(window, self.iqr_multiplier)
        if point.value < lower or point.value > upper:
            return Anomaly(
                metric_name=point.name,
                value=point.value,
                method="iqr",
                score=max(abs(point.value - upper), abs(point.value - lower)),
                threshold=self.iqr_multiplier,
                message=f"Value {point.value} outside IQR bounds [{lower:.2f}, {upper:.2f}]",
                timestamp=point.timestamp,
                labels=point.labels,
            )
        return None

    def detect(self, point: MetricPoint) -> list[Anomaly]:
        """Run all detection methods and return any anomalies found.

        The window is updated once per call so that z-score and IQR
        operate on the same state.
        """
        window = self._update_window(point.name, point.value)
        anomalies: list[Anomaly] = []
        for checker in (
            self._check_threshold,
            lambda p: self._check_zscore(p, window),
            lambda p: self._check_iqr(p, window),
        ):
            result = checker(point)
            if result is not None:
                anomalies.append(result)
        return anomalies

    def detect_batch(self, points: list[MetricPoint]) -> list[Anomaly]:
        """Process a batch of metric points and return all anomalies."""
        all_anomalies: list[Anomaly] = []
        for point in points:
            all_anomalies.extend(self.detect(point))
        return all_anomalies


def format_anomaly_report(anomalies: list[Anomaly]) -> str:
    """Format anomalies into a human-readable report."""
    if not anomalies:
        return "No anomalies detected."
    lines = [f"Detected {len(anomalies)} anomaly(ies):", ""]
    for a in anomalies:
        lines.append(
            f"  [{a.method.upper()}] {a.metric_name}: "
            f"value={a.value}, score={a.score}, threshold={a.threshold}"
        )
        lines.append(f"    {a.message}")
        lines.append("")
    return "\n".join(lines)
