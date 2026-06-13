"""Tests for the anomaly detection module."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detector import (
    Anomaly,
    AnomalyDetector,
    MetricPoint,
    format_anomaly_report,
    iqr_bounds,
    mean,
    stdev,
    z_score,
)


class TestStatisticalFunctions:
    def test_mean_empty(self) -> None:
        assert mean([]) == 0.0

    def test_mean_single(self) -> None:
        assert mean([5.0]) == 5.0

    def test_mean_multiple(self) -> None:
        assert mean([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0

    def test_stdev_empty(self) -> None:
        assert stdev([]) == 0.0

    def test_stdev_single(self) -> None:
        assert stdev([5.0]) == 0.0

    def test_stdev_uniform(self) -> None:
        assert stdev([5.0, 5.0, 5.0]) == 0.0

    def test_stdev_known_values(self) -> None:
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        s = stdev(vals)
        assert abs(s - 2.0) < 0.01  # known stdev is 2.0

    def test_zscore(self) -> None:
        assert z_score(10.0, 5.0, 2.5) == 2.0
        assert z_score(0.0, 5.0, 2.5) == -2.0

    def test_zscore_zero_stdev(self) -> None:
        assert z_score(10.0, 10.0, 0.0) == 0.0

    def test_iqr_bounds_basic(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        lower, upper = iqr_bounds(vals, multiplier=1.5)
        assert lower < 1.0
        assert upper > 8.0

    def test_iqr_bounds_too_few(self) -> None:
        lower, upper = iqr_bounds([1.0, 2.0])
        assert lower == float("-inf")
        assert upper == float("inf")


class TestMetricPoint:
    def test_create_point(self) -> None:
        p = MetricPoint(name="cpu_usage", value=85.5, timestamp=1000.0, labels={"host": "web-1"})
        assert p.name == "cpu_usage"
        assert p.value == 85.5

    def test_point_to_dict(self) -> None:
        p = MetricPoint(name="mem", value=42.0)
        d = p.to_dict()
        assert d["name"] == "mem"
        assert d["value"] == 42.0
        assert d["labels"] == {}


class TestAnomaly:
    def test_create_anomaly(self) -> None:
        a = Anomaly(
            metric_name="cpu",
            value=99.0,
            method="threshold",
            score=1.1,
            threshold=90.0,
            message="Too high",
        )
        assert a.metric_name == "cpu"
        assert a.method == "threshold"

    def test_anomaly_to_dict(self) -> None:
        a = Anomaly(
            metric_name="rps",
            value=5000.0,
            method="zscore",
            score=4.5,
            threshold=3.0,
            message="Z-score spike",
        )
        d = a.to_dict()
        assert d["method"] == "zscore"
        assert d["score"] == 4.5


class TestAnomalyDetector:
    def test_zscore_no_anomaly_normal_data(self) -> None:
        detector = AnomalyDetector(zscore_threshold=3.0, window_size=20)
        # Feed consistent data
        for i in range(10):
            point = MetricPoint(name="test", value=50.0 + (i % 2))
            anomalies = detector.detect(point)
        assert anomalies == []

    def test_zscore_detects_spike(self) -> None:
        detector = AnomalyDetector(zscore_threshold=2.0, window_size=20)
        # Feed normal data
        for i in range(10):
            detector.detect(MetricPoint(name="test", value=50.0))
        # Spike
        anomalies = detector.detect(MetricPoint(name="test", value=200.0))
        zscore_anomalies = [a for a in anomalies if a.method == "zscore"]
        assert len(zscore_anomalies) == 1
        assert zscore_anomalies[0].value == 200.0

    def test_zscore_needs_minimum_data(self) -> None:
        detector = AnomalyDetector(zscore_threshold=1.0)
        # Only 2 points - not enough for z-score
        detector.detect(MetricPoint(name="test", value=10.0))
        anomalies = detector.detect(MetricPoint(name="test", value=1000.0))
        zscore_anomalies = [a for a in anomalies if a.method == "zscore"]
        assert len(zscore_anomalies) == 0

    def test_threshold_detects_breach(self) -> None:
        detector = AnomalyDetector(threshold_rules={"error_rate": 5.0})
        anomalies = detector.detect(MetricPoint(name="error_rate", value=10.0))
        threshold_anomalies = [a for a in anomalies if a.method == "threshold"]
        assert len(threshold_anomalies) == 1
        assert threshold_anomalies[0].threshold == 5.0

    def test_threshold_no_breach(self) -> None:
        detector = AnomalyDetector(threshold_rules={"error_rate": 5.0})
        anomalies = detector.detect(MetricPoint(name="error_rate", value=3.0))
        threshold_anomalies = [a for a in anomalies if a.method == "threshold"]
        assert len(threshold_anomalies) == 0

    def test_threshold_no_rule(self) -> None:
        detector = AnomalyDetector(threshold_rules={"error_rate": 5.0})
        anomalies = detector.detect(MetricPoint(name="other_metric", value=100.0))
        threshold_anomalies = [a for a in anomalies if a.method == "threshold"]
        assert len(threshold_anomalies) == 0

    def test_iqr_detects_outlier(self) -> None:
        detector = AnomalyDetector(window_size=50, iqr_multiplier=1.5)
        # Feed normal data (need at least 4 for IQR)
        for _ in range(10):
            detector.detect(MetricPoint(name="test", value=50.0))
        # Outlier
        anomalies = detector.detect(MetricPoint(name="test", value=500.0))
        iqr_anomalies = [a for a in anomalies if a.method == "iqr"]
        assert len(iqr_anomalies) == 1

    def test_iqr_needs_minimum_data(self) -> None:
        detector = AnomalyDetector(window_size=50)
        for i in range(3):
            detector.detect(MetricPoint(name="test", value=float(i)))
        anomalies = detector.detect(MetricPoint(name="test", value=1000.0))
        iqr_anomalies = [a for a in anomalies if a.method == "iqr"]
        assert len(iqr_anomalies) == 0

    def test_detect_batch(self) -> None:
        detector = AnomalyDetector(
            zscore_threshold=2.0,
            threshold_rules={"errors": 10.0},
            window_size=50,
        )
        points = [MetricPoint(name="cpu", value=50.0) for _ in range(10)]
        points.append(MetricPoint(name="cpu", value=500.0))  # spike
        points.append(MetricPoint(name="errors", value=50.0))  # threshold breach
        anomalies = detector.detect_batch(points)
        assert len(anomalies) >= 2

    def test_multiple_metrics_independent(self) -> None:
        import random

        random.seed(42)
        detector = AnomalyDetector(zscore_threshold=3.0, window_size=20)
        # Feed variable data so small deviations don't look anomalous
        for _ in range(10):
            detector.detect(MetricPoint(name="cpu", value=50.0 + random.uniform(-5, 5)))
            detector.detect(MetricPoint(name="mem", value=80.0 + random.uniform(-5, 5)))
        # Spike only in cpu
        cpu_anomalies = detector.detect(MetricPoint(name="cpu", value=200.0))
        mem_anomalies = detector.detect(MetricPoint(name="mem", value=82.0))
        assert any(a.method == "zscore" for a in cpu_anomalies)
        assert all(a.method != "zscore" for a in mem_anomalies)

    def test_window_size_limit(self) -> None:
        detector = AnomalyDetector(zscore_threshold=3.0, window_size=5)
        for i in range(20):
            detector.detect(MetricPoint(name="test", value=float(i)))
        assert len(detector._windows["test"]) == 5


class TestFormatAnomalyReport:
    def test_empty_report(self) -> None:
        assert format_anomaly_report([]) == "No anomalies detected."

    def test_report_with_anomalies(self) -> None:
        anomalies = [
            Anomaly(
                metric_name="cpu",
                value=99.0,
                method="zscore",
                score=4.5,
                threshold=3.0,
                message="Spike detected",
            ),
            Anomaly(
                metric_name="errors",
                value=50.0,
                method="threshold",
                score=5.0,
                threshold=10.0,
                message="Error rate too high",
            ),
        ]
        report = format_anomaly_report(anomalies)
        assert "2 anomaly" in report
        assert "cpu" in report
        assert "ZSCORE" in report
        assert "THRESHOLD" in report
        assert "Spike detected" in report
