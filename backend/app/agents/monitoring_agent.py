"""Monitoring Agent (Phase 5): is a deployed champion still seeing data like it was trained on?

Reads the Prediction Playground / API request log (`PredictionLog`) for a finished run and
compares what is being asked of the model against the feature schema recorded at training
time (numeric ranges, known categories). It also watches the model's own confidence over
time. Read-only and advisory: it *recommends* retraining, a human decides.

Signals (per feature): out-of-range rate for numeric inputs (beyond the training range plus a
10% margin), unseen-category rate for categorical inputs, missing-value rate. Model-level:
drop in mean predicted confidence between the earlier and later half of the window.
"""
from __future__ import annotations

MIN_LOGS = 20
RANGE_MARGIN = 0.10
WARN_RATE, CRIT_RATE = 0.10, 0.30
WARN_CONF_DROP, CRIT_CONF_DROP = 0.10, 0.20


def _feature_report(name: str, spec: dict, values: list) -> dict:
    n = len(values)
    missing = sum(1 for v in values if v in (None, ""))
    present = [v for v in values if v not in (None, "")]
    report = {"feature": name, "type": spec.get("type"), "missing_rate": round(missing / n, 3) if n else 0.0}
    if spec.get("type") == "numeric":
        lo, hi = spec.get("min"), spec.get("max")
        out = 0
        if lo is not None and hi is not None:
            margin = (hi - lo) * RANGE_MARGIN
            for v in present:
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    out += 1
                    continue
                if x < lo - margin or x > hi + margin:
                    out += 1
        report["drift_rate"] = round(out / len(present), 3) if present else 0.0
        report["drift_kind"] = "out_of_training_range"
    else:
        known = {str(o) for o in spec.get("options", [])}
        unseen = sum(1 for v in present if str(v) not in known)
        report["drift_rate"] = round(unseen / len(present), 3) if present else 0.0
        report["drift_kind"] = "unseen_category"
    return report


def assess(feature_schema: dict, logs: list[dict]) -> dict:
    """`logs`: oldest-first list of {"input": {...}, "output": {...}} dicts."""
    n = len(logs)
    if n < MIN_LOGS:
        return {
            "status": "insufficient_data", "n_predictions": n, "min_required": MIN_LOGS,
            "recommend_retrain": False, "findings": [],
            "summary": f"Only {n} logged prediction(s); at least {MIN_LOGS} are needed for a meaningful drift check.",
        }

    features = [
        _feature_report(name, spec, [(entry.get("input") or {}).get(name) for entry in logs])
        for name, spec in feature_schema.items()
    ]

    confidences = [
        (entry.get("output") or {}).get("confidence") for entry in logs
        if isinstance((entry.get("output") or {}).get("confidence"), (int, float))
    ]
    conf_drop = None
    if len(confidences) >= MIN_LOGS:
        half = len(confidences) // 2
        conf_drop = round(sum(confidences[:half]) / half - sum(confidences[half:]) / (len(confidences) - half), 3)

    findings: list[dict] = []
    status = "ok"

    def escalate(level: str) -> None:
        nonlocal status
        if level == "critical" or (level == "warning" and status == "ok"):
            status = level

    for f in features:
        if f["drift_rate"] >= CRIT_RATE:
            escalate("critical")
            findings.append({"severity": "critical", "feature": f["feature"],
                             "message": f"{f['drift_rate']:.0%} of recent inputs for '{f['feature']}' fall outside what the model was trained on ({f['drift_kind']})."})
        elif f["drift_rate"] >= WARN_RATE:
            escalate("warning")
            findings.append({"severity": "warning", "feature": f["feature"],
                             "message": f"{f['drift_rate']:.0%} of recent inputs for '{f['feature']}' look unfamiliar ({f['drift_kind']})."})
    if conf_drop is not None:
        if conf_drop >= CRIT_CONF_DROP:
            escalate("critical")
            findings.append({"severity": "critical", "feature": None, "message": f"The model's average confidence fell by {conf_drop:.0%} between the earlier and later predictions."})
        elif conf_drop >= WARN_CONF_DROP:
            escalate("warning")
            findings.append({"severity": "warning", "feature": None, "message": f"The model's average confidence fell by {conf_drop:.0%} between the earlier and later predictions."})

    return {
        "status": status,
        "n_predictions": n,
        "recommend_retrain": status != "ok",
        "confidence_drop": conf_drop,
        "features": sorted(features, key=lambda f: f["drift_rate"], reverse=True),
        "findings": findings,
        "summary": (
            "Recent inputs look like the training data." if status == "ok"
            else f"{len(findings)} drift signal(s) found — consider retraining on fresh data."
        ),
    }
