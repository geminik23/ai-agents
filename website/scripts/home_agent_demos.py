#!/usr/bin/env python3
"""Record public fixed-response conversations through the normal agent runtime.

Normal website builds only check source hashes. --record runs credential-free
mocked scenarios; the exported record contains only the declared public exchange.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib

WEBSITE = Path(__file__).resolve().parents[1]
ROOT = WEBSITE.parent
CATALOG = WEBSITE / "catalog/home-agents.toml"
RECORD = WEBSITE / "catalog/home-conversations.json"


def source_hashes(agents):
    paths = [CATALOG]
    for agent in agents:
        path = (WEBSITE / agent["yaml_path"]).resolve()
        if not path.is_relative_to((WEBSITE / "static/agents").resolve()):
            raise ValueError("Homepage agents must use public static/agents YAML")
        paths.append(path)
    return {str(path.relative_to(WEBSITE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def record(agents, hashes):
    """Exercise each complete YAML and allowlist its verified, public response."""
    output = ROOT / "target/website-demo/home-agents"
    output.mkdir(parents=True, exist_ok=True)
    environment = {key: value for key, value in os.environ.items() if not key.endswith("_API_KEY")}
    records = []
    for agent in agents:
        suite = {"name": f"Homepage {agent['name']}", "agent": str(WEBSITE / agent["yaml_path"]),
                 "settings": {"retries": 0, "redact_outputs": False},
                 "fixtures": {"llm": {"mode": "mock", "responses": [agent["response"]]}},
                 "scenarios": [{"id": agent["id"], "turns": [{"input": agent["input"],
                    "assert": {"all": [{"response_not_empty": True}, {"response_contains": agent["response"]},
                                        {"tool_not_called": "http"}, {"tool_not_called": "command"}]}}]}]}
        suite_path = output / f"{agent['id']}.yaml"
        suite_path.write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
        report_dir = output / agent["id"]
        subprocess.run(["cargo", "run", "-p", "ai-agents-cli", "--locked", "--offline", "--", "eval",
                        "--scenarios", str(suite_path), "--output", str(report_dir)],
                       cwd=ROOT, env=environment, check=True)
        report = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        scenario, = report["scenarios"]
        attempt, = scenario["attempts"]
        turn, = attempt["turns"]
        if report["passed"] != 1 or not turn["response_present"] or turn["response"]["value"] != agent["response"]:
            raise ValueError("The runtime did not return the declared public demo response")
        records.append({"id": agent["id"], "input": agent["input"], "response": turn["response"]["value"]})
    result = {"mode": "mock", "source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "source_hashes": hashes, "agents": records}
    RECORD.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check(agents, hashes):
    recorded = json.loads(RECORD.read_text(encoding="utf-8"))
    expected = [{key: agent[key] for key in ("id", "input", "response")} for agent in agents]
    if recorded["mode"] != "mock" or recorded["source_hashes"] != hashes or recorded["agents"] != expected:
        raise ValueError("Homepage demo sources changed; run python3 website/scripts/home_agent_demos.py --record")
    print("Homepage conversation records match their public YAML and fixtures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    agents = tomllib.loads(CATALOG.read_text(encoding="utf-8"))["agents"]
    hashes = source_hashes(agents)
    if args.record:
        record(agents, hashes)
    check(agents, hashes)
