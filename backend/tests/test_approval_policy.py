from __future__ import annotations

from app.services import approval_policy_service


def test_auto_approves_high_confidence_no_risk():
    auto, reason = approval_policy_service.decide("transformation", {}, 0.9)
    assert auto is True
    assert "auto-approved" in reason.lower()


def test_requires_review_when_confidence_below_threshold():
    auto, reason = approval_policy_service.decide("transformation", {}, 0.5)
    assert auto is False
    assert "confidence" in reason.lower()


def test_requires_review_when_confidence_missing():
    auto, reason = approval_policy_service.decide("transformation", {}, None)
    assert auto is False


def test_data_validation_escalates_on_critical_check_even_with_high_confidence():
    decision = {"checks": [{"name": "Data leakage risk", "status": "critical", "detail": "x", "affected_columns": ["c"]}]}
    auto, reason = approval_policy_service.decide("data_validation", decision, 0.95)
    assert auto is False
    assert "leakage" in reason.lower()


def test_data_validation_auto_approves_when_all_checks_ok():
    decision = {"checks": [{"name": "Missing values", "status": "ok", "detail": "x", "affected_columns": []}]}
    auto, reason = approval_policy_service.decide("data_validation", decision, 0.95)
    assert auto is True


def test_cleaning_plan_escalates_on_no_information_column():
    decision = {"recommendations": [{"column": "x", "action": "drop_column", "no_information": True}]}
    auto, reason = approval_policy_service.decide("cleaning_plan", decision, 0.9)
    assert auto is False
    assert "no usable information" in reason.lower()


def test_cleaning_plan_escalates_on_leakage_driven_drop():
    decision = {"recommendations": [{"column": "sqft", "action": "drop_column", "issue": "data leakage risk", "no_information": False}]}
    auto, reason = approval_policy_service.decide("cleaning_plan", decision, 0.9)
    assert auto is False
    assert "leakage" in reason.lower()


def test_cleaning_plan_escalates_on_large_missing_target():
    decision = {"recommendations": [], "target_missing": {"column": "y", "missing_pct": 35.0}}
    auto, reason = approval_policy_service.decide("cleaning_plan", decision, 0.9)
    assert auto is False


def test_cleaning_plan_auto_approves_clean_plan():
    decision = {"recommendations": [{"column": "x", "action": "mean", "no_information": False}]}
    auto, reason = approval_policy_service.decide("cleaning_plan", decision, 0.9)
    assert auto is True


def test_mandatory_reason_is_stable_text():
    assert "always requires human review" in approval_policy_service.mandatory_reason().lower()
