"""
Part 1 - ML Tool: predict_churn(features) + compare_scenarios(features, changes)

Follows the team Tool Contract (data/shared_schema.json) and NEVER raises:
  success -> {"status": "ok",    "tool": "predict_churn", "data": {...}, "error": None}
  failure -> {"status": "error", "tool": "predict_churn", "data": None,
              "error": {"type": "...", "message": "...", "missing_fields": [...]}}
Error types allowed by the contract: validation_error | not_found |
connection_error | internal_error  (extra keys `field` / `allowed_values` are added
on validation errors to help the agent ask the user the right question).

Allowed values, required fields, tenure groups, risk levels all come from
data/shared_schema.json -> one source of truth for the whole team.
"""
import json
import math

import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH, SHARED_SCHEMA_PATH
from ml_model.features import ADDONS, FEATURES, add_features

TOOL_NAME = "predict_churn"
META_PATH = MODEL_PATH.with_name("model_meta.json")   # decision threshold saved by train.py

_model = None    # lazy-loaded once
_schema = None   # lazy-loaded once

# Optional fields the user may skip -> filled with these and reported in data["assumed_defaults"].
# (The 8 required fields from the schema have NO default: the agent must ask for them.)
DEFAULTS = {
    "gender": "Male", "Partner": "No", "Dependents": "No",
    "PhoneService": "Yes", "MultipleLines": "No",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
}

# tenure_group name -> segment name (names must match shared_schema.json "segments")
TENURE_SEGMENT = {"0-12": "New Customer", "13-24": "Growing Customer",
                  "25-48": "Established Customer", "49-72": "Loyal Customer"}


class _ToolError(Exception):
    """Internal only: carries a contract-shaped error out of the helpers."""
    def __init__(self, err_type, message, **extra):
        self.err_type, self.message, self.extra = err_type, message, extra


# --------------------------------------------------------------------------- #
# Contract helpers
# --------------------------------------------------------------------------- #
def _ok(data: dict) -> dict:
    return {"status": "ok", "tool": TOOL_NAME, "data": data, "error": None}


def _err(err_type: str, message: str, missing_fields=None, **extra) -> dict:
    return {"status": "error", "tool": TOOL_NAME, "data": None,
            "error": {"type": err_type, "message": message,
                      "missing_fields": missing_fields or [], **extra}}


# --------------------------------------------------------------------------- #
# Schema / model loading
# --------------------------------------------------------------------------- #
def _get_schema() -> dict:
    global _schema
    if _schema is None:
        _schema = json.loads(SHARED_SCHEMA_PATH.read_text(encoding="utf-8"))
    return _schema


def allowed_values() -> dict:
    """{field: [allowed values]} for every categorical field (read from the schema)."""
    fields = _get_schema()["ml_features"]["fields"]
    return {k: v["values"] for k, v in fields.items() if v["type"] == "category"}


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def _get_threshold() -> float:
    """Decision threshold tuned in train.py (falls back to 0.5)."""
    try:
        return float(json.loads(META_PATH.read_text()).get("threshold", 0.5))
    except Exception:
        return 0.5


def _risk_level(p: float) -> str:
    """Bands come from schema risk_levels: Low [0,0.3) / Medium [0.3,0.6) / High [0.6,1]."""
    levels = _get_schema()["risk_levels"]
    for lvl in levels[:-1]:
        if p < lvl["max_probability"]:
            return lvl["name"]
    return levels[-1]["name"]


def _tenure_group(months: float) -> str:
    for g in _get_schema()["tenure_groups"]:
        if g["min_months"] <= months <= g["max_months"]:
            return g["name"]
    raise _ToolError("validation_error", f"tenure {months} is outside the allowed range.")


def _segments(tenure_group: str, monthly: float, senior: int) -> list:
    segs = [TENURE_SEGMENT[tenure_group]]
    if monthly >= 80:
        segs.append("High Spender")
    if senior == 1:
        segs.append("Senior Customer")
    return segs


# --------------------------------------------------------------------------- #
# Explanation helpers
# --------------------------------------------------------------------------- #
def _pretty(name: str) -> str:
    """'cat__Contract_Month-to-month' -> 'Contract = Month-to-month' (readable for the agent)."""
    if name.startswith("cat__"):
        col, _, val = name[5:].partition("_")
        return f"{col} = {val}"
    name = name[5:] if name.startswith("num__") else name
    if "^2" in name:
        return f"{name.replace('^2', '')} (squared)"
    if " " in name:
        return name.replace(" ", " x ") + " (interaction)"
    return name


def _top_factors(model, X, n_up=3, n_down=2) -> dict:
    """Per-customer explanation: coefficient x transformed value (linear model)."""
    prep, clf = model.named_steps["prep"], model.named_steps["model"]
    Z = prep.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    contrib = Z[0] * clf.coef_[0]
    names = [_pretty(n) for n in prep.get_feature_names_out()]
    order = np.argsort(contrib)
    up = [names[i] for i in order[::-1][:n_up] if contrib[i] > 0]
    down = [names[i] for i in order[:n_down] if contrib[i] < 0]
    return {"increase_churn_risk": up, "decrease_churn_risk": down}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _to_number(row: dict, key: str) -> float:
    v = row[key]
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise _ToolError("validation_error", f"{key} must be a number.", field=key)
    try:
        v = float(v)
    except ValueError:
        raise _ToolError("validation_error", f"{key} must be a number.", field=key)
    if not math.isfinite(v):
        raise _ToolError("validation_error", f"{key} must be a finite number.", field=key)
    if v < 0:
        raise _ToolError("validation_error", f"{key} must be >= 0.", field=key)
    return v


def _validate_and_build_row(features: dict):
    """Returns (row, assumed_defaults, adjustments) or raises _ToolError."""
    required = _get_schema()["ml_features"]["required"]
    missing = [k for k in required if features.get(k) in (None, "")]
    if missing:
        raise _ToolError("validation_error",
                         f"Missing required fields: {', '.join(missing)}",
                         missing_fields=missing)

    assumed = [k for k in DEFAULTS if features.get(k) is None]
    row = {**DEFAULTS, **{k: v for k, v in features.items() if v is not None}}

    # ---- numbers ----
    tenure = _to_number(row, "tenure")
    if tenure != int(tenure) or tenure > 72:
        raise _ToolError("validation_error",
                         "tenure must be a whole number of months between 0 and 72.",
                         field="tenure")
    monthly = _to_number(row, "MonthlyCharges")
    senior = _to_number(row, "SeniorCitizen")
    if senior not in (0.0, 1.0):
        raise _ToolError("validation_error", "SeniorCitizen must be 0 or 1.",
                         field="SeniorCitizen", allowed_values=[0, 1])
    if row.get("TotalCharges") in (None, ""):
        total = tenure * monthly            # estimate when the user doesn't know it
        assumed.append("TotalCharges")
    else:
        total = _to_number(row, "TotalCharges")
    row.update(tenure=tenure, MonthlyCharges=monthly, SeniorCitizen=int(senior),
               TotalCharges=total)

    # ---- business-rule consistency ----
    adjustments = []
    if row["InternetService"] == "No":
        for k in ADDONS:
            if row[k] != "No internet service":
                adjustments.append(f"{k} set to 'No internet service' (customer has no internet)")
            row[k] = "No internet service"
    if row["PhoneService"] == "No" and row["MultipleLines"] != "No phone service":
        adjustments.append("MultipleLines set to 'No phone service' (customer has no phone)")
        row["MultipleLines"] = "No phone service"

    # ---- categories ----
    for col, allowed in allowed_values().items():
        if row[col] not in allowed:
            raise _ToolError("validation_error", f"Invalid value for {col}: {row[col]!r}",
                             field=col, allowed_values=allowed)
    if row["InternetService"] != "No":
        for k in ADDONS:
            if row[k] == "No internet service":
                raise _ToolError("validation_error",
                                 f"{k} cannot be 'No internet service' when the customer has "
                                 f"{row['InternetService']} internet.",
                                 field=k, allowed_values=["Yes", "No"])
    if row["PhoneService"] == "Yes" and row["MultipleLines"] == "No phone service":
        raise _ToolError("validation_error",
                         "MultipleLines cannot be 'No phone service' when PhoneService is Yes.",
                         field="MultipleLines", allowed_values=["Yes", "No"])
    return row, assumed, adjustments


# --------------------------------------------------------------------------- #
# The tool
# --------------------------------------------------------------------------- #
def predict_churn(features: dict) -> dict:
    """Predict whether a telecom customer will churn.

    Required keys (from shared_schema.json): tenure (0-72 months), Contract,
    InternetService, PaymentMethod, MonthlyCharges, SeniorCitizen (0/1),
    TechSupport, PaperlessBilling.
    Optional: gender, Partner, Dependents, PhoneService, MultipleLines,
    OnlineSecurity, OnlineBackup, DeviceProtection, StreamingTV, StreamingMovies,
    TotalCharges. Skipped optional fields get defaults and are listed in
    data["assumed_defaults"] so the agent knows what was assumed.
    """
    try:
        if not isinstance(features, dict):
            return _err("validation_error", "Input must be a JSON object (dict).")
        row, assumed, adjustments = _validate_and_build_row(features)

        model = _get_model()
        X = add_features(pd.DataFrame([row]))[FEATURES]
        prob = float(model.predict_proba(X)[0][1])
        thr = _get_threshold()
        group = _tenure_group(row["tenure"])

        return _ok({
            "churn_prediction": "Yes" if prob >= thr else "No",
            "churn_probability": round(prob, 3),
            "risk_level": _risk_level(prob),
            "threshold_used": thr,
            "customer_profile": {
                "contract": row["Contract"],
                "internet_service": row["InternetService"],
                "tenure_group": group,
                "payment_method": row["PaymentMethod"],
                "monthly_charges": row["MonthlyCharges"],
                "senior_citizen": row["SeniorCitizen"],
                "has_tech_support": row["TechSupport"],
                "segments": _segments(group, row["MonthlyCharges"], row["SeniorCitizen"]),
            },
            "top_factors": _top_factors(model, X),
            "assumed_defaults": assumed,
            "adjustments": adjustments,
        })
    except _ToolError as e:
        return _err(e.err_type, e.message, **e.extra)
    except FileNotFoundError as e:
        return _err("not_found", f"Required file not found ({e.filename}). "
                                 "Run: python -m ml_model.train")
    except Exception as e:  # never crash the agent
        return _err("internal_error", f"Unexpected error: {e}")


def compare_scenarios(features: dict, changes: dict) -> dict:
    """What-if helper: predict the same customer with and without `changes`.

    Example: compare_scenarios(customer, {"Contract": "One year"})
    Returns data = {"base": {...}, "scenario": {...}, "changes": {...},
                    "probability_change": scenario - base}.
    """
    try:
        if not isinstance(features, dict) or not isinstance(changes, dict) or not changes:
            return _err("validation_error",
                        "features and changes must both be non-empty JSON objects (dict).")
        base = predict_churn(features)
        if base["status"] != "ok":
            return base
        scen = predict_churn({**features, **changes})
        if scen["status"] != "ok":
            return scen
        b, s = base["data"], scen["data"]
        return _ok({
            "base": b, "scenario": s, "changes": changes,
            "probability_change": round(s["churn_probability"] - b["churn_probability"], 3),
        })
    except Exception as e:
        return _err("internal_error", f"Unexpected error: {e}")


if __name__ == "__main__":
    demo = {"tenure": 2, "MonthlyCharges": 70.7, "Contract": "Month-to-month",
            "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
            "SeniorCitizen": 0, "TechSupport": "No", "PaperlessBilling": "Yes"}
    print(json.dumps(predict_churn(demo), indent=2, ensure_ascii=False))
    print(json.dumps(compare_scenarios(demo, {"Contract": "One year"}), indent=2,
                     ensure_ascii=False))
