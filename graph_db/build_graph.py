import argparse
import json
import sys

from neo4j import GraphDatabase

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, SHARED_SCHEMA_PATH

LABELS = ["Segment", "Plan", "Offer", "EligibilityCondition", "Contract"]


def load_schema() -> dict:
    return json.loads(SHARED_SCHEMA_PATH.read_text(encoding="utf-8"))


def _check_schema_consistency(schema: dict) -> None:
    """Fail early (with a clear message) if the schema references an undefined name.
    This catches typos like 'Fiber Plus ' or 'New Customers' BEFORE they silently
    create a missing edge in the graph."""
    segs = {s["name"] for s in schema["segments"]}
    plans = {p["name"] for p in schema["plans"]}
    conds = {c["name"] for c in schema["eligibility_conditions"]}
    contracts = set(schema["contracts"])
    problems = []
    for p in schema["plans"]:
        for c in p["available_contracts"]:
            if c not in contracts:
                problems.append(f"plan '{p['name']}' -> unknown contract '{c}'")
    for o in schema["offers"]:
        for s in o["target_segments"]:
            if s not in segs:
                problems.append(f"offer '{o['name']}' -> unknown segment '{s}'")
        for p in o["applicable_plans"]:
            if p not in plans:
                problems.append(f"offer '{o['name']}' -> unknown plan '{p}'")
        for c in o["requires"]:
            if c not in conds:
                problems.append(f"offer '{o['name']}' -> unknown condition '{c}'")
    if problems:
        raise ValueError("shared_schema.json is inconsistent:\n  - " + "\n  - ".join(problems))


def create_constraints(session) -> None:
    for label in LABELS:
        session.run(
            f"CREATE CONSTRAINT {label.lower()}_name_unique IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
        )


def reset_graph(session) -> None:
    session.run("MATCH (n) DETACH DELETE n")


def build_nodes(session, schema: dict) -> None:
    session.run(
        "UNWIND $rows AS r MERGE (s:Segment {name: r.name}) SET s.rule = r.rule",
        rows=schema["segments"],
    )
    session.run(
        "UNWIND $rows AS r MERGE (c:Contract {name: r})",
        rows=schema["contracts"],
    )
    session.run(
        "UNWIND $rows AS r MERGE (p:Plan {name: r.name}) "
        "SET p.internet_service = r.internet_service, p.description = r.description",
        rows=schema["plans"],
    )
    session.run(
        "UNWIND $rows AS r MERGE (e:EligibilityCondition {name: r.name}) SET e.rule = r.rule",
        rows=schema["eligibility_conditions"],
    )
    session.run(
        "UNWIND $rows AS r MERGE (o:Offer {name: r.name}) SET o.value = r.value",
        rows=schema["offers"],
    )


def build_relationships(session, schema: dict) -> None:
    plan_contract = [
        {"plan": p["name"], "contract": c}
        for p in schema["plans"] for c in p["available_contracts"]
    ]
    session.run(
        "UNWIND $rows AS r "
        "MATCH (p:Plan {name: r.plan}), (c:Contract {name: r.contract}) "
        "MERGE (p)-[:AVAILABLE_WITH]->(c)",
        rows=plan_contract,
    )
    targets = [{"offer": o["name"], "target": s} for o in schema["offers"] for s in o["target_segments"]]
    session.run(
        "UNWIND $rows AS r MATCH (o:Offer {name: r.offer}), (s:Segment {name: r.target}) "
        "MERGE (o)-[:TARGETS]->(s)",
        rows=targets,
    )
    applies = [{"offer": o["name"], "plan": p} for o in schema["offers"] for p in o["applicable_plans"]]
    session.run(
        "UNWIND $rows AS r MATCH (o:Offer {name: r.offer}), (p:Plan {name: r.plan}) "
        "MERGE (o)-[:APPLIES_TO]->(p)",
        rows=applies,
    )
    requires = [{"offer": o["name"], "cond": c} for o in schema["offers"] for c in o["requires"]]
    session.run(
        "UNWIND $rows AS r MATCH (o:Offer {name: r.offer}), (e:EligibilityCondition {name: r.cond}) "
        "MERGE (o)-[:REQUIRES]->(e)",
        rows=requires,
    )


def graph_counts(session) -> dict:
    nodes = {
        row["label"]: row["n"]
        for row in session.run(
            "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY label"
        )
    }
    rels = {
        row["t"]: row["n"]
        for row in session.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY t")
    }
    return {"nodes": nodes, "relationships": rels}


def expected_counts(schema: dict) -> dict:
    """What the graph SHOULD contain, computed from the schema (used by tests + the final check)."""
    return {
        "nodes": {
            "Contract": len(schema["contracts"]),
            "EligibilityCondition": len(schema["eligibility_conditions"]),
            "Offer": len(schema["offers"]),
            "Plan": len(schema["plans"]),
            "Segment": len(schema["segments"]),
        },
        "relationships": {
            "APPLIES_TO": sum(len(o["applicable_plans"]) for o in schema["offers"]),
            "AVAILABLE_WITH": sum(len(p["available_contracts"]) for p in schema["plans"]),
            "REQUIRES": sum(len(o["requires"]) for o in schema["offers"]),
            "TARGETS": sum(len(o["target_segments"]) for o in schema["offers"]),
        },
    }


def build_graph(reset: bool = False) -> dict:
    """Build the whole graph. Returns the real counts found in Neo4j afterwards."""
    schema = load_schema()
    _check_schema_consistency(schema)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            if reset:
                reset_graph(session)
            create_constraints(session)
            build_nodes(session, schema)
            build_relationships(session, schema)
            return graph_counts(session)
    finally:
        driver.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the TeleNova retention graph in Neo4j.")
    parser.add_argument("--reset", action="store_true", help="delete all existing nodes first")
    args = parser.parse_args()
    try:
        counts = build_graph(reset=args.reset)
    except ValueError as e:                      # schema problem
        print(f"[schema error] {e}")
        return 2
    except Exception as e:                       # Neo4j down / wrong password
        print(f"[connection error] Could not build the graph on {NEO4J_URI}: {e}")
        print("Is Neo4j running?  ->  docker compose up -d")
        return 1
    expected = expected_counts(load_schema())
    print("Graph built.")
    print(json.dumps(counts, indent=2))
    print("Matches schema:", counts == expected)
    return 0 if counts == expected else 3


if __name__ == "__main__":
    sys.exit(main())