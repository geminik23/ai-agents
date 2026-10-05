#!/usr/bin/env python3
"""Publish a small allowlisted projection of the public HITL mocked eval report.

Run the suite with the current checkout first. Full eval evidence is intentionally
not serialized by the runtime; this walkthrough presents verified assertions and
the public fixture's three cases, not a raw trace or live provider measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SUITE = "examples/eval/mocked/hitl/hitl_basic_mocked.yaml"
AGENT = "examples/yaml/hitl/hitl_basic.yaml"
DESTINATION = ROOT / "website/catalog/execution-demo.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(report_path):
    """Fail closed unless all three expected turns have passing structural checks."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (ROOT / report["agent"]).resolve() != (ROOT / AGENT).resolve():
        raise ValueError("The report does not use the public HITL example")
    if report["suite"] != "HITL Basic Mocked Eval" or report["passed"] != 1 or report["failed"] != 0:
        raise ValueError("Expected a passing HITL Basic Mocked Eval report")
    scenario, = report["scenarios"]
    if scenario["id"] != "approve-reject-timeout" or scenario["status"] != "passed":
        raise ValueError("Unexpected scenario or status")
    attempt, = scenario["attempts"]
    if len(attempt["turns"]) != 3:
        raise ValueError("Expected exactly three recorded turns")
    cases = [
        ("approved", "Approve", True, "Approved request", "The approval fixture accepts the request. The HTTP tool fixture executes and returns status 200."),
        ("rejected", "Reject", False, "Rejected request", "The approval fixture rejects the request. The tool execution check confirms that the HTTP implementation did not run."),
        ("timeout", "Time out", False, "Approval timeout", "The approval fixture returns a timeout. The configured timeout policy rejects the request, and the HTTP implementation does not run."),
    ]
    turns = []
    for index, (turn, case) in enumerate(zip(attempt["turns"], cases)):
        key, label, executed, title, description = case
        checks = turn["assertion_results"]
        if turn["index"] != index or not turn["response_present"] or not checks or not all(check["passed"] for check in checks):
            raise ValueError("A turn is missing its finalized response or passing checks")
        execution = [check for check in checks if check["assertion"] == "tool_called"
                     and isinstance(check["expected"], dict) and check["expected"].get("executed") is executed
                     and check["expected"].get("success") is executed
                     and check["expected"].get("count") == 1 and check["actual"] == 1]
        approval = [check for check in checks if check["assertion"] == "approval_requested"
                    and check["actual"] == {"matched_count": 1, "total_count": 1}]
        if len(execution) != 1 or not approval:
            raise ValueError("Missing execution or approval evidence")
        if executed and execution[0]["expected"]["result_path"].get("eq") != 200:
            raise ValueError("Approved fixture must return status 200")
        turns.append({"id": key, "label": label, "title": title, "description": description,
                      "executed": executed, "success": executed, "checks_passed": len(checks),
                      "approval_matches": approval[0]["actual"]["matched_count"],
                      "assertions": [{"name": check["assertion"], "passed": check["passed"]} for check in checks]})
    artifact = {"mode": "mock", "scenario": scenario["id"], "source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_suite": SUITE, "source_agent": AGENT,
        "suite_sha256": digest(ROOT / SUITE), "agent_sha256": digest(ROOT / AGENT),
        "report_sha256": digest(report_path), "turns": turns}
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print("Captured three assertion-backed mocked outcomes")


def check():
    """Keep recorded evidence tied to the public fixture and agent it documents."""
    artifact = json.loads(DESTINATION.read_text(encoding="utf-8"))
    if artifact["mode"] != "mock" or artifact["source_suite"] != SUITE or artifact["source_agent"] != AGENT:
        raise ValueError("Unexpected execution demo provenance")
    if artifact["suite_sha256"] != digest(ROOT / SUITE) or artifact["agent_sha256"] != digest(ROOT / AGENT):
        raise ValueError("Execution demo sources changed; rerun the public mocked suite and recapture its report")
    if [turn["id"] for turn in artifact["turns"]] != ["approved", "rejected", "timeout"]:
        raise ValueError("Execution demo is missing a recorded outcome")
    for turn in artifact["turns"]:
        if not turn["assertions"] or not all(item["passed"] for item in turn["assertions"]):
            raise ValueError("Execution demo contains a failing assertion")
    print("Execution demo source hashes match")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?", help="summary.json from the public HITL mocked suite")
    parser.add_argument("--check", action="store_true", help="Verify the recorded public source hashes without executing an agent")
    args = parser.parse_args()
    if args.check:
        check()
    elif args.report:
        capture(args.report)
    else:
        parser.error("provide a report or --check")
