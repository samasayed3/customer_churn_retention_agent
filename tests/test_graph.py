import json

import pytest
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, SHARED_SCHEMA_PATH
from graph_db import build_graph as bg
from graph_db import query_graph as qg
from graph_db.query_graph import infer_plans, query_retention_graph

SCHEMA = json.loads(SHARED_SCHEMA_PATH.read_text(encoding="utf-8"))
ERROR_TYPES = {"validation_error", "not_found", "connection_error", "internal_error"}


def assert_contract(res: dict):
    """Every return value must follow the team Tool Contract."""
    assert res["tool"] == "query_retention_graph"
    assert res["status"] in ("ok", "error")
    if res["status"] == "ok":
        assert isinstance(res["data"], dict) and res["error"] is None
    else:
        assert res["data"] is None
        assert res["error"]["type"] in ERROR_TYPES
        assert isinstance(res["error"]["message"], str)
        assert isinstance(res["error"]["missing_fields"], list)


# =========================================================================== #
# 1) UNIT TESTS (no Neo4j needed)
# =========================================================================== #
def test_schema_is_internally_consistent():
    bg._check_schema_consistency(SCHEMA)          # raises ValueError on any unknown name


def test_schema_inconsistency_is_detected():
    bad = json.loads(json.dumps(SCHEMA))
    bad["offers"][0]["target_segments"].append("New Customers")   # typo
    with pytest.raises(ValueError, match="New Customers"):
        bg._check_schema_consistency(bad)


def test_expected_counts_from_schema():
    exp = bg.expected_counts(SCHEMA)
    assert exp["nodes"] == {"Contract": 3, "EligibilityCondition": 9, "Offer": 7,
                            "Plan": 4, "Segment": 6}
    assert exp["relationships"]["AVAILABLE_WITH"] == 10
    assert exp["relationships"]["TARGETS"] == 14
    assert exp["relationships"]["APPLIES_TO"] == 20
    assert exp["relationships"]["REQUIRES"] == 9


def test_all_schema_query_types_are_implemented():
    assert set(SCHEMA["graph_query_types"]) == set(qg._HANDLERS)
    for qt, spec in SCHEMA["graph_query_types"].items():
        for p in qg.QUERY_TYPES[qt]:
            assert p in spec["params"]


def test_unknown_query_type():
    res = query_retention_graph("run_cypher", {"q": "MATCH (n) DETACH DELETE n"})
    assert_contract(res)
    assert res["error"]["type"] == "validation_error"
    assert "offers_for_segment" in res["error"]["allowed_values"]


@pytest.mark.parametrize("qt,params,missing", [
    ("offers_for_segment", {}, ["segment"]),
    ("offers_for_plan", {"plan": ""}, ["plan"]),
    ("eligibility_for_offer", {"offer": None}, ["offer"]),
    ("plans_by_contract", {}, ["contract"]),
    ("offers_for_profile", {"plan": "Basic DSL"}, ["segments"]),
    ("offers_for_profile", {"segments": []}, ["segments"]),
])
def test_missing_params(qt, params, missing):
    res = query_retention_graph(qt, params)
    assert_contract(res)
    assert res["error"]["type"] == "validation_error"
    assert res["error"]["missing_fields"] == missing


def test_params_not_a_dict():
    res = query_retention_graph("offers_for_segment", "New Customer")
    assert_contract(res)
    assert res["error"]["type"] == "validation_error"


def test_wrong_name_returns_not_found_with_allowed_values():
    res = query_retention_graph("offers_for_segment", {"segment": "VIP Customer"})
    assert_contract(res)
    assert res["error"]["type"] == "not_found"
    assert "New Customer" in res["error"]["allowed_values"]


def test_wrong_name_inside_segments_list():
    res = query_retention_graph("offers_for_profile", {"segments": ["New Customer", "Nope"]})
    assert_contract(res)
    assert res["error"]["type"] == "not_found"


def test_names_are_case_insensitive(monkeypatch):
    seen = {}
    monkeypatch.setattr(qg, "_run", lambda cypher, **p: seen.update(p) or [])
    monkeypatch.setattr(qg, "_graph_is_built", lambda: True)
    res = query_retention_graph("offers_for_segment", {"segment": "  new customer "})
    assert_contract(res)
    assert seen["segment"] == "New Customer"          # canonical name goes to Neo4j


def test_neo4j_down_returns_connection_error_not_exception(monkeypatch):
    def boom(*a, **k):
        raise ServiceUnavailable("down")
    monkeypatch.setattr(qg, "_run", boom)
    res = query_retention_graph("plans_by_contract", {"contract": "Two year"})
    assert_contract(res)
    assert res["error"]["type"] == "connection_error"


def test_empty_graph_is_reported(monkeypatch):
    monkeypatch.setattr(qg, "_run", lambda c, **p: [])
    monkeypatch.setattr(qg, "_graph_is_built", lambda: False)
    res = query_retention_graph("offers_for_segment", {"segment": "Loyal Customer"})
    assert_contract(res)
    assert res["error"]["type"] == "internal_error"
    assert "build_graph" in res["error"]["message"]


def test_unexpected_exception_never_escapes(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("weird")
    monkeypatch.setattr(qg, "_run", boom)
    res = query_retention_graph("offers_for_plan", {"plan": "Fiber Plus"})
    assert_contract(res)
    assert res["error"]["type"] == "internal_error"


def test_no_cypher_is_built_from_user_input(monkeypatch):
    """Input only ever travels as a $parameter, never inside the Cypher text."""
    captured = {}
    monkeypatch.setattr(qg, "_run", lambda cypher, **p: captured.update(cypher=cypher, p=p) or [])
    monkeypatch.setattr(qg, "_graph_is_built", lambda: True)
    query_retention_graph("offers_for_segment", {"segment": "New Customer"})
    assert "New Customer" not in captured["cypher"]
    assert captured["p"] == {"segment": "New Customer"}


# ---- the "missing plan" gap: predict_churn returns NO plan -------------------- #
@pytest.mark.parametrize("profile,expected", [
    ({"internet_service": "DSL", "contract": "Month-to-month", "has_tech_support": "No"}, ["Basic DSL"]),
    ({"internet_service": "Fiber optic", "contract": "Month-to-month", "has_tech_support": "No"}, ["Fiber Plus"]),
    ({"internet_service": "Fiber optic", "contract": "One year", "has_tech_support": "No"}, ["Fiber Plus"]),
    ({"internet_service": "Fiber optic", "contract": "Two year", "has_tech_support": "Yes"},
     ["Fiber Plus", "Premium Bundle"]),
    ({"internet_service": "No", "contract": "Month-to-month", "has_tech_support": "No internet service"},
     ["Phone Only"]),
    ({"internet_service": "No", "contract": "Two year", "has_tech_support": "No internet service"}, []),
])
def test_infer_plans(profile, expected):
    assert infer_plans(profile) == expected


def test_plan_inferred_from_real_predict_churn_output(monkeypatch):
    """End-to-end (no Neo4j): real ML output -> profile -> inferred plan -> Cypher params."""
    try:
        from ml_model.predict import predict_churn
    except ImportError as e:                       # numpy / pandas / sklearn not installed
        pytest.skip(f"ML dependencies not installed: {e}")
    res = predict_churn({"tenure": 2, "MonthlyCharges": 70.7, "Contract": "Month-to-month",
                         "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
                         "SeniorCitizen": 0, "TechSupport": "No", "PaperlessBilling": "Yes"})
    if res["status"] != "ok":                      # e.g. model.joblib saved with another sklearn version
        pytest.skip(f"predict_churn not usable in this environment: {res['error']['message']}")
    profile = res["data"]["customer_profile"]
    assert "plan" not in profile                      # <- the gap
    captured = {}
    monkeypatch.setattr(qg, "_run", lambda c, **p: captured.update(p) or [])
    monkeypatch.setattr(qg, "_graph_is_built", lambda: True)
    out = query_retention_graph("offers_for_profile",
                                {"segments": profile["segments"], "customer_profile": profile})
    assert_contract(out)
    assert captured["plans"] == ["Fiber Plus"]
    assert out["data"]["plan_inferred"] is True


def test_profile_without_plan_info_does_not_filter(monkeypatch):
    captured = {}
    monkeypatch.setattr(qg, "_run", lambda c, **p: captured.update(p) or [])
    monkeypatch.setattr(qg, "_graph_is_built", lambda: True)
    out = query_retention_graph("offers_for_profile", {"segments": ["New Customer"]})
    assert_contract(out)
    assert captured["plans"] == []
    assert out["data"]["plans_considered"] is None


# =========================================================================== #
# 2) INTEGRATION TESTS (need Neo4j: docker compose up -d)
# =========================================================================== #
def _neo4j_up() -> bool:
    try:
        d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), connection_timeout=2)
        d.verify_connectivity()
        d.close()
        return True
    except Exception:
        return False


needs_neo4j = pytest.mark.skipif(not _neo4j_up(), reason="Neo4j is not running (docker compose up -d)")


@pytest.fixture(scope="module")
def built_graph():
    counts = bg.build_graph(reset=True)
    yield counts
    qg.close_driver()


def _offers(**flt):
    return [o for o in SCHEMA["offers"] if all(flt[k](o) for k in flt)]


@needs_neo4j
def test_graph_matches_schema(built_graph):
    assert built_graph == bg.expected_counts(SCHEMA)


@needs_neo4j
def test_build_is_idempotent(built_graph):
    assert bg.build_graph(reset=False) == bg.expected_counts(SCHEMA)


@needs_neo4j
@pytest.mark.parametrize("segment", [s["name"] for s in SCHEMA["segments"]])
def test_offers_for_segment(built_graph, segment):
    res = query_retention_graph("offers_for_segment", {"segment": segment})
    assert_contract(res)
    got = sorted(o["name"] for o in res["data"]["offers"])
    assert got == sorted(o["name"] for o in _offers(t=lambda o: segment in o["target_segments"]))


@needs_neo4j
@pytest.mark.parametrize("plan", [p["name"] for p in SCHEMA["plans"]])
def test_offers_for_plan(built_graph, plan):
    res = query_retention_graph("offers_for_plan", {"plan": plan})
    assert_contract(res)
    got = sorted(o["name"] for o in res["data"]["offers"])
    assert got == sorted(o["name"] for o in _offers(a=lambda o: plan in o["applicable_plans"]))


@needs_neo4j
@pytest.mark.parametrize("offer", [o["name"] for o in SCHEMA["offers"]])
def test_eligibility_for_offer(built_graph, offer):
    res = query_retention_graph("eligibility_for_offer", {"offer": offer})
    assert_contract(res)
    expected = next(o for o in SCHEMA["offers"] if o["name"] == offer)["requires"]
    assert [c["name"] for c in res["data"]["conditions"]] == sorted(expected)
    assert all(c["rule"] for c in res["data"]["conditions"])


@needs_neo4j
@pytest.mark.parametrize("contract", SCHEMA["contracts"])
def test_plans_by_contract(built_graph, contract):
    res = query_retention_graph("plans_by_contract", {"contract": contract})
    assert_contract(res)
    expected = sorted(p["name"] for p in SCHEMA["plans"] if contract in p["available_contracts"])
    assert [p["name"] for p in res["data"]["plans"]] == expected


@needs_neo4j
def test_offers_for_profile_with_plan(built_graph):
    segs, plan = ["Senior Customer", "Loyal Customer"], "Basic DSL"
    res = query_retention_graph("offers_for_profile", {"segments": segs, "plan": plan})
    assert_contract(res)
    expected = sorted(o["name"] for o in SCHEMA["offers"]
                      if set(o["target_segments"]) & set(segs) and plan in o["applicable_plans"])
    assert sorted(o["name"] for o in res["data"]["offers"]) == expected
    for o in res["data"]["offers"]:
        assert o["eligibility_conditions"]            # every offer carries its conditions


@needs_neo4j
def test_offers_for_profile_inferred_plan_ambiguous(built_graph):
    profile = {"internet_service": "Fiber optic", "contract": "Two year", "has_tech_support": "Yes"}
    res = query_retention_graph("offers_for_profile",
                                {"segments": ["High Spender"], "customer_profile": profile})
    assert_contract(res)
    assert res["data"]["plans_considered"] == ["Fiber Plus", "Premium Bundle"]
    assert [o["name"] for o in res["data"]["offers"]] == ["Contract Upgrade Discount", "Fiber Price Lock"]