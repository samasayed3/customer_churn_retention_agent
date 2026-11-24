"""Standalone tests for the ML tool (no agent needed).  Run from the repo root:
    python -m pytest tests/test_ml.py -v
"""
import json
import math

import pytest

from ml_model.predict import allowed_values, compare_scenarios, predict_churn

# the 8 required fields from data/shared_schema.json
CUSTOMER = {
    "tenure": 3, "MonthlyCharges": 80, "Contract": "Month-to-month",
    "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
    "SeniorCitizen": 0, "TechSupport": "No", "PaperlessBilling": "Yes",
}
LOYAL = {
    "tenure": 60, "MonthlyCharges": 55, "Contract": "Two year",
    "InternetService": "DSL", "PaymentMethod": "Bank transfer (automatic)",
    "SeniorCitizen": 0, "TechSupport": "Yes", "PaperlessBilling": "No",
}
ERROR_TYPES = {"validation_error", "not_found", "connection_error", "internal_error"}


def assert_error(res, err_type=None):
    """Every error must follow the Tool Contract."""
    assert res["status"] == "error"
    assert res["tool"] == "predict_churn"
    assert res["data"] is None
    assert isinstance(res["error"], dict)
    assert res["error"]["type"] in ERROR_TYPES
    assert res["error"]["message"]
    assert isinstance(res["error"]["missing_fields"], list)
    if err_type:
        assert res["error"]["type"] == err_type
    json.dumps(res)  # must be JSON-serialisable


# ---------------------------- success ---------------------------- #
def test_success_follows_contract():
    res = predict_churn(CUSTOMER)
    assert res["status"] == "ok" and res["tool"] == "predict_churn" and res["error"] is None
    d = res["data"]
    assert d["churn_prediction"] in ("Yes", "No")
    assert 0.0 <= d["churn_probability"] <= 1.0
    assert d["risk_level"] in ("Low", "Medium", "High")
    p = d["customer_profile"]
    assert set(p) == {"contract", "internet_service", "tenure_group", "payment_method",
                      "monthly_charges", "senior_citizen", "has_tech_support", "segments"}
    assert p["contract"] == "Month-to-month"
    assert p["tenure_group"] == "0-12"
    assert p["monthly_charges"] == 80.0 and p["senior_citizen"] == 0
    assert p["has_tech_support"] == "No"
    json.dumps(res)


def test_assumed_defaults_are_reported():
    d = predict_churn(CUSTOMER)["data"]
    assert "gender" in d["assumed_defaults"] and "TotalCharges" in d["assumed_defaults"]
    full = predict_churn({**CUSTOMER, "gender": "Female"})["data"]
    assert "gender" not in full["assumed_defaults"]


@pytest.mark.parametrize("months,group", [(0, "0-12"), (12, "0-12"), (13, "13-24"),
                                          (24, "13-24"), (25, "25-48"), (48, "25-48"),
                                          (49, "49-72"), (72, "49-72")])
def test_tenure_groups(months, group):
    res = predict_churn({**CUSTOMER, "tenure": months})
    assert res["data"]["customer_profile"]["tenure_group"] == group


def test_segments():
    segs = predict_churn({**CUSTOMER, "SeniorCitizen": 1})["data"]["customer_profile"]["segments"]
    assert set(segs) == {"New Customer", "High Spender", "Senior Customer"}
    segs = predict_churn(LOYAL)["data"]["customer_profile"]["segments"]
    assert segs == ["Loyal Customer"]


def test_risk_level_bands():
    # same bands as shared_schema.json: Low < 0.3 <= Medium < 0.6 <= High
    for c in (CUSTOMER, LOYAL):
        d = predict_churn(c)["data"]
        p, lvl = d["churn_probability"], d["risk_level"]
        assert lvl == ("Low" if p < 0.3 else "Medium" if p < 0.6 else "High")


def test_high_vs_low_risk_ordering():
    risky = predict_churn(CUSTOMER)["data"]
    loyal = predict_churn(LOYAL)["data"]
    assert risky["churn_probability"] > loyal["churn_probability"]
    assert risky["risk_level"] == "High" and loyal["risk_level"] == "Low"
    assert loyal["churn_prediction"] == "No"


def test_no_internet_is_consistent():
    res = predict_churn({**CUSTOMER, "InternetService": "No", "TechSupport": "Yes",
                         "MonthlyCharges": 20, "Contract": "One year"})
    assert res["status"] == "ok"                       # contradictory input is auto-fixed
    assert res["data"]["customer_profile"]["has_tech_support"] == "No internet service"
    assert res["data"]["adjustments"]


# ---------------------------- errors ---------------------------- #
def test_missing_fields():
    res = predict_churn({"tenure": 5})
    assert_error(res, "validation_error")
    assert set(res["error"]["missing_fields"]) == {
        "Contract", "InternetService", "PaymentMethod", "MonthlyCharges",
        "SeniorCitizen", "TechSupport", "PaperlessBilling"}


def test_senior_citizen_zero_is_not_missing():
    assert predict_churn({**CUSTOMER, "SeniorCitizen": 0})["status"] == "ok"


def test_invalid_category():
    res = predict_churn({**CUSTOMER, "Contract": "Weekly"})
    assert_error(res, "validation_error")
    assert res["error"]["allowed_values"] == allowed_values()["Contract"]


def test_invalid_optional_category():
    assert_error(predict_churn({**CUSTOMER, "gender": "Robot"}), "validation_error")


@pytest.mark.parametrize("field", ["tenure", "MonthlyCharges", "TotalCharges"])
def test_negative_values_rejected(field):
    assert_error(predict_churn({**CUSTOMER, field: -5}), "validation_error")


@pytest.mark.parametrize("tenure", [73, 100, 2.5])
def test_tenure_out_of_range_or_fractional(tenure):
    assert_error(predict_churn({**CUSTOMER, "tenure": tenure}), "validation_error")


def test_non_numeric():
    assert_error(predict_churn({**CUSTOMER, "tenure": "abc"}), "validation_error")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_nan_inf_bool_rejected(bad):
    assert_error(predict_churn({**CUSTOMER, "MonthlyCharges": bad}))


def test_bad_senior_citizen():
    assert_error(predict_churn({**CUSTOMER, "SeniorCitizen": 2}), "validation_error")


def test_contradictory_addon_with_internet():
    assert_error(predict_churn({**CUSTOMER, "TechSupport": "No internet service"}),
                 "validation_error")


@pytest.mark.parametrize("bad", ["hello", None, 42, [1, 2], {}])
def test_non_dict_or_empty_never_raises(bad):
    assert_error(predict_churn(bad))


# ------------------------- scenario comparison ------------------------- #
def test_compare_monthly_vs_one_year():
    res = compare_scenarios(CUSTOMER, {"Contract": "One year"})
    assert res["status"] == "ok"
    d = res["data"]
    assert d["base"]["customer_profile"]["contract"] == "Month-to-month"
    assert d["scenario"]["customer_profile"]["contract"] == "One year"
    assert d["scenario"]["churn_probability"] < d["base"]["churn_probability"]
    assert math.isclose(d["probability_change"],
                        d["scenario"]["churn_probability"] - d["base"]["churn_probability"],
                        abs_tol=0.002)


def test_compare_does_not_mutate_input():
    original = dict(CUSTOMER)
    compare_scenarios(CUSTOMER, {"Contract": "Two year"})
    assert CUSTOMER == original


def test_compare_propagates_errors():
    assert_error(compare_scenarios(CUSTOMER, {"Contract": "Weekly"}), "validation_error")
    assert_error(compare_scenarios({"tenure": 1}, {"Contract": "One year"}), "validation_error")
    assert_error(compare_scenarios(CUSTOMER, {}), "validation_error")
