#!/usr/bin/env python3
"""CLI entry point: detect anomalies in application metrics using statistical and AI methods."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .detector import (
    Anomaly,
    AnomalyDetector,
    MetricPoint,
    format_anomaly_report,
)
from .llm import LLMError, call_llm

SYSTEM_PROMPT = """\
You are an SRE anomaly analysis assistant. Given a set of detected metric anomalies, \
provide:
1. A root-cause hypothesis for each anomaly.
2. Recommended immediate actions.
3. Whether these anomalies likely indicate a single correlated incident.

Return a JSON object with this schema:
{
  "analysis": [
    {
      "metric": "string",
      "hypothesis": "string",
      "confidence": "high|medium|low",
      "recommended_actions": ["string"]
    }
  ],
  "correlated": true|false,
  "summary": "string"
}
Return ONLY valid JSON.
"""


def load_metrics_from_json(raw: str) -> list[MetricPoint]:
    """Parse a JSON array of metric points."""
    data = json.loads(raw)
    points: list[MetricPoint] = []
    for item in data:
        points.append(
            MetricPoint(
                name=item.get("name", "unknown"),
                value=float(item.get("value", 0)),
                timestamp=float(item.get("timestamp", 0)),
                labels=item.get("labels", {}),
            )
        )
    return points


def build_analysis_prompt(anomalies: list[Anomaly]) -> str:
    """Build LLM prompt from detected anomalies."""
    anomaly_dicts = [a.to_dict() for a in anomalies]
    return (
        "Analyze the following detected metric anomalies:\n\n"
        f"{json.dumps(anomaly_dicts, indent=2)}\n\n"
        "Provide root-cause hypotheses and recommended actions."
    )


def analyze_with_llm(anomalies: list[Anomaly], model: str = "llama3") -> str:
    """Send anomalies to LLM for root-cause analysis."""
    prompt = build_analysis_prompt(anomalies)
    return call_llm(prompt, model=model, system=SYSTEM_PROMPT)


def run_detection(
    points: list[MetricPoint],
    zscore_threshold: float = 3.0,
    window_size: int = 50,
    threshold_rules: dict[str, float] | None = None,
) -> list[Anomaly]:
    """Run anomaly detection on a batch of metric points."""
    detector = AnomalyDetector(
        zscore_threshold=zscore_threshold,
        window_size=window_size,
        threshold_rules=threshold_rules,
    )
    return detector.detect_batch(points)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sidecar container that detects anomalies in application metrics."
    )
    parser.add_argument(
        "-i", "--input",
        help="Path to metrics JSON file (array of {name, value, timestamp, labels}).",
    )
    parser.add_argument(
        "-m", "--model", default="llama3",
        help="Ollama model for AI analysis (default: llama3).",
    )
    parser.add_argument(
        "--zscore", type=float, default=3.0,
        help="Z-score threshold (default: 3.0).",
    )
    parser.add_argument(
        "--window", type=int, default=50,
        help="Rolling window size (default: 50).",
    )
    parser.add_argument(
        "--threshold", action="append", default=[],
        help="Static threshold rule as 'metric_name=value' (repeatable).",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Send detected anomalies to LLM for root-cause analysis.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Write output to file instead of stdout.",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default: text).",
    )
    args = parser.parse_args(argv)

    # Parse threshold rules
    threshold_rules: dict[str, float] = {}
    for rule in args.threshold:
        if "=" in rule:
            name, val = rule.split("=", 1)
            threshold_rules[name.strip()] = float(val.strip())

    # Load metrics
    if args.input:
        try:
            with open(args.input) as fh:
                raw = fh.read()
        except FileNotFoundError:
            print(f"Error: file not found: {args.input}", file=sys.stderr)
            return 1
    else:
        raw = sys.stdin.read()

    try:
        points = load_metrics_from_json(raw)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"Error parsing metrics: {exc}", file=sys.stderr)
        return 1

    # Run detection
    anomalies = run_detection(
        points,
        zscore_threshold=args.zscore,
        window_size=args.window,
        threshold_rules=threshold_rules,
    )

    # Format output
    if args.format == "json":
        result_data: dict[str, object] = {
            "anomalies": [a.to_dict() for a in anomalies],
            "total_points": len(points),
            "total_anomalies": len(anomalies),
        }
        if args.analyze and anomalies:
            try:
                llm_result = analyze_with_llm(anomalies, model=args.model)
                result_data["llm_analysis"] = json.loads(llm_result)
            except Exception as exc:
                result_data["llm_analysis_error"] = str(exc)
        result = json.dumps(result_data, indent=2)
    else:
        result = format_anomaly_report(anomalies)
        if args.analyze and anomalies:
            try:
                llm_result = analyze_with_llm(anomalies, model=args.model)
                result += f"\n\nAI Analysis:\n{llm_result}\n"
            except Exception as exc:
                result += f"\n\nAI Analysis Error: {exc}\n"

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(result)
    else:
        print(result)

    return 1 if anomalies else 0


if __name__ == "__main__":
    raise SystemExit(main())
