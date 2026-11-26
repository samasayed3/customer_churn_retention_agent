import json
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, SHARED_SCHEMA_PATH

TOOL_NAME = "query_retention_graph"

_driver = None
_schema = None


# --------------------------------------------------------------------------- #
# Contract helpers
# --------------------------------------------------------------------------- #
class _ToolError(Exception):
    """Internal only: carries a contract-shaped error out of the helpers."""
    def __init__(self, err_type: str, message: str, missing_fields=None, **extra):
        self.err_type, self.message = err_type, message
        self.missing_fields, self.extra = missing_fields or [], extra


def _ok(data: dict) -> dict:
    return {"status": "ok", "tool": TOOL_NAME, "data": data, "error": None}


def _err(err_type: str, message: str, missing_fields=None, **extra) -> dict:
    return {"status": "error", "tool": TOOL_NAME, "data": None,
            "error": {"type": err_type, "message": message,
                      "missing_fields": missing_fields or [], **extra}}


# --------------------------------------------------------------------------- #
# Schema (used only to validate / normalise names, never as the data source of results)
# --------------------------------------------------------------------------- #
def _get_schema() -> dict:
    global _schema
    if _schema is None:
        _schema = json.loads(SHARED_SCHEMA_PATH.read_text(encoding="utf-8"))
    return _schema


def _names(kind: str) -> list:
    s = _get_schema()
    if kind == "segment":
        return [x["name"] for x in s["segments"]]
    if kind == "plan":
        return [x["name"] for x in s["plans"]]
    if kind == "offer":
        return [x["name"] for x in s["offers"]]
    if kind == "contract":
        return list(s["contracts"])
    raise KeyError(kind)


def _resolve(kind: str, value: Any, param: str) -> str:
    """Return the canonical name (case/space-insensitive match) or raise a contract error."""
    if not isinstance(value, str) or not value.strip():
        raise _ToolError("validation_error", f"'{param}' must be a non-empty string.",
                         missing_fields=[param], allowed_values=_names(kind))
    wanted = value.strip().lower()
    for name in _names(kind):
        if name.lower() == wanted:
            return name
    raise _ToolError("not_found", f"Unknown {kind} {value!r}. Use one of the official names.",
                     allowed_values=_names(kind))


# --------------------------------------------------------------------------- #
# Plan inference (fills the gap: predict_churn does not return a plan)
# --------------------------------------------------------------------------- #
def infer_plans(customer_profile: dict) -> list:
    """Candidate plans for a predict_churn `customer_profile`, from schema facts only:
         * plan.internet_service must equal the customer's internet_service
         * the customer's contract must be in plan.available_contracts
         * Premium Bundle INCLUDES tech support -> only a candidate if has_tech_support == "Yes"
       Returns a list (0, 1 or 2 names). Two candidates = genuinely ambiguous
       (Fiber + One/Two year + tech support: Fiber Plus or Premium Bundle).
    """
    internet = customer_profile.get("internet_service")
    contract = customer_profile.get("contract")
    tech = customer_profile.get("has_tech_support")
    out = []
    for p in _get_schema()["plans"]:
        if p["internet_service"] != internet:
            continue
        if contract not in p["available_contracts"]:
            continue
        if p["name"] == "Premium Bundle" and tech != "Yes":
            continue
        out.append(p["name"])
    return out


# --------------------------------------------------------------------------- #
# Neo4j access
# --------------------------------------------------------------------------- #
def _get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
                                       connection_timeout=3, max_transaction_retry_time=2)
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def _run(cypher: str, **params) -> list:
    """Run one fixed Cypher template. Returns a list of plain dict rows."""
    with _get_driver().session() as session:
        return [dict(r) for r in session.run(cypher, **params)]


def _graph_is_built() -> bool:
    rows = _run("MATCH (o:Offer) RETURN count(o) AS n")
    return bool(rows and rows[0]["n"] > 0)


_COND = ("collect(DISTINCT CASE WHEN e IS NULL THEN null "
         "ELSE {name: e.name, rule: e.rule} END)")

# --------------------------------------------------------------------------- #
# Fixed Cypher templates (no LLM-written Cypher)
# --------------------------------------------------------------------------- #
CYPHER = {
    "offers_for_segment": """
        MATCH (o:Offer)-[:TARGETS]->(:Segment {name: $segment})
        OPTIONAL MATCH (o)-[:APPLIES_TO]->(p:Plan)
        WITH o, collect(DISTINCT p.name) AS plans
        OPTIONAL MATCH (o)-[:REQUIRES]->(e:EligibilityCondition)
        RETURN o.name AS name, o.value AS value, plans AS applicable_plans,
               collect(DISTINCT e.name) AS requires
        ORDER BY name
    """,
    "offers_for_plan": """
        MATCH (o:Offer)-[:APPLIES_TO]->(:Plan {name: $plan})
        OPTIONAL MATCH (o)-[:TARGETS]->(s:Segment)
        WITH o, collect(DISTINCT s.name) AS segs
        OPTIONAL MATCH (o)-[:REQUIRES]->(e:EligibilityCondition)
        RETURN o.name AS name, o.value AS value, segs AS target_segments,
               collect(DISTINCT e.name) AS requires
        ORDER BY name
    """,
    "eligibility_for_offer": """
        MATCH (o:Offer {name: $offer})
        OPTIONAL MATCH (o)-[:REQUIRES]->(e:EligibilityCondition)
        RETURN o.name AS name, o.value AS value, %s AS conditions
    """ % _COND,
    "plans_by_contract": """
        MATCH (p:Plan)-[:AVAILABLE_WITH]->(:Contract {name: $contract})
        RETURN p.name AS name, p.internet_service AS internet_service,
               p.description AS description
        ORDER BY name
    """,
    "offers_for_profile": """
        MATCH (o:Offer)-[:TARGETS]->(s:Segment)
        WHERE s.name IN $segments
        WITH o, collect(DISTINCT s.name) AS matched
        MATCH (o)-[:APPLIES_TO]->(p:Plan)
        WHERE size($plans) = 0 OR p.name IN $plans
        WITH o, matched, collect(DISTINCT p.name) AS plans
        OPTIONAL MATCH (o)-[:REQUIRES]->(e:EligibilityCondition)
        RETURN o.name AS name, o.value AS value, matched AS matched_segments,
               plans AS applicable_plans, %s AS eligibility_conditions
        ORDER BY size(matched) DESC, name
    """ % _COND,
}

QUERY_TYPES = {
    "offers_for_segment": ["segment"],
    "offers_for_plan": ["plan"],
    "eligibility_for_offer": ["offer"],
    "plans_by_contract": ["contract"],
    "offers_for_profile": ["segments"],
}


def _sorted(rows: list, *list_keys: str) -> list:
    for r in rows:
        for k in list_keys:
            r[k] = sorted(r[k], key=lambda x: x["name"] if isinstance(x, dict) else x)
    return rows


# --------------------------------------------------------------------------- #
# One handler per query_type
# --------------------------------------------------------------------------- #
def _q_offers_for_segment(params: dict) -> dict:
    seg = _resolve("segment", params.get("segment"), "segment")
    rows = _sorted(_run(CYPHER["offers_for_segment"], segment=seg), "applicable_plans", "requires")
    return {"segment": seg, "offers": rows, "count": len(rows)}


def _q_offers_for_plan(params: dict) -> dict:
    plan = _resolve("plan", params.get("plan"), "plan")
    rows = _sorted(_run(CYPHER["offers_for_plan"], plan=plan), "target_segments", "requires")
    return {"plan": plan, "offers": rows, "count": len(rows)}


def _q_eligibility_for_offer(params: dict) -> dict:
    offer = _resolve("offer", params.get("offer"), "offer")
    rows = _run(CYPHER["eligibility_for_offer"], offer=offer)
    if not rows:
        raise _ToolError("internal_error",
                         f"Offer {offer!r} is not in the graph. Build it: python -m graph_db.build_graph")
    r = rows[0]
    return {"offer": r["name"], "value": r["value"],
            "conditions": sorted(r["conditions"], key=lambda c: c["name"])}


def _q_plans_by_contract(params: dict) -> dict:
    contract = _resolve("contract", params.get("contract"), "contract")
    rows = _run(CYPHER["plans_by_contract"], contract=contract)
    return {"contract": contract, "plans": rows, "count": len(rows)}


def _q_offers_for_profile(params: dict) -> dict:
    segs_in = params.get("segments")
    if isinstance(segs_in, str):
        segs_in = [segs_in]
    if not isinstance(segs_in, list) or not segs_in:
        raise _ToolError("validation_error", "'segments' must be a non-empty list of segment names.",
                         missing_fields=["segments"], allowed_values=_names("segment"))
    segments = sorted({_resolve("segment", s, "segments") for s in segs_in})

    plan_in, warning, inferred = params.get("plan"), None, False
    if plan_in in (None, "", []):
        profile = params.get("customer_profile")
        if isinstance(profile, dict):
            plans = infer_plans(profile)
            inferred = True
            if not plans:
                warning = ("No plan matches this contract + internet service combination; "
                           "offers were NOT filtered by plan.")
        else:
            plans = []          # no plan info at all -> do not filter by plan
    else:
        if isinstance(plan_in, str):
            plan_in = [plan_in]
        if not isinstance(plan_in, list):
            raise _ToolError("validation_error", "'plan' must be a string or a list of plan names.",
                             allowed_values=_names("plan"))
        plans = sorted({_resolve("plan", p, "plan") for p in plan_in})

    rows = _sorted(_run(CYPHER["offers_for_profile"], segments=segments, plans=plans),
                   "matched_segments", "applicable_plans")
    for r in rows:
        r["eligibility_conditions"] = sorted(r["eligibility_conditions"], key=lambda c: c["name"])
    data = {"segments": segments, "plans_considered": plans or None, "plan_inferred": inferred,
            "offers": rows, "count": len(rows)}
    if len(plans) > 1:
        data["note"] = "Plan is ambiguous; offers shown for any of the candidate plans."
    if warning:
        data["warning"] = warning
    return data


_HANDLERS = {
    "offers_for_segment": _q_offers_for_segment,
    "offers_for_plan": _q_offers_for_plan,
    "eligibility_for_offer": _q_eligibility_for_offer,
    "plans_by_contract": _q_plans_by_contract,
    "offers_for_profile": _q_offers_for_profile,
}


# --------------------------------------------------------------------------- #
# The tool
# --------------------------------------------------------------------------- #
def query_retention_graph(query_type: str, params: dict) -> dict:
    """Query the TeleNova retention graph (offers, plans, segments, eligibility).

    query_type is ONE of: offers_for_segment, offers_for_plan, eligibility_for_offer,
    plans_by_contract, offers_for_profile. `params` holds the parameters of that type
    (see module docstring). Names must be the official ones from shared_schema.json
    (case-insensitive). Never raises; always returns a contract-shaped dict.
    """
    try:
        if not isinstance(query_type, str) or query_type not in _HANDLERS:
            return _err("validation_error",
                        f"Unknown query_type {query_type!r}.",
                        allowed_values=sorted(_HANDLERS))
        if not isinstance(params, dict):
            return _err("validation_error", "params must be a JSON object (dict).",
                        missing_fields=QUERY_TYPES[query_type])
        missing = [k for k in QUERY_TYPES[query_type] if params.get(k) in (None, "", [])]
        if missing:
            return _err("validation_error",
                        f"Missing required params for {query_type}: {', '.join(missing)}",
                        missing_fields=missing)

        data = _HANDLERS[query_type](params)

        # An empty answer is only trustworthy if the graph really exists.
        if data.get("count") == 0 and not _graph_is_built():
            return _err("internal_error",
                        "The graph is empty. Build it first: python -m graph_db.build_graph")
        return _ok(data)

    except _ToolError as e:
        return _err(e.err_type, e.message, e.missing_fields, **e.extra)
    except (ServiceUnavailable, AuthError, OSError) as e:
        return _err("connection_error",
                    f"Cannot reach Neo4j at {NEO4J_URI} ({type(e).__name__}). "
                    "Check that it is running: docker compose up -d")
    except Neo4jError as e:
        return _err("internal_error", f"Neo4j error: {e.message}")
    except Exception as e:  # never crash the agent
        return _err("internal_error", f"Unexpected error: {e}")


if __name__ == "__main__":
    demo = [
        ("offers_for_segment", {"segment": "New Customer"}),
        ("offers_for_profile", {"segments": ["Senior Customer", "Loyal Customer"], "plan": "Basic DSL"}),
        ("eligibility_for_offer", {"offer": "Fiber Price Lock"}),
        ("plans_by_contract", {"contract": "Two year"}),
    ]
    for qt, p in demo:
        print(qt, p)
        print(json.dumps(query_retention_graph(qt, p), indent=2, ensure_ascii=False))