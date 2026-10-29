# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# ─── How to run ───
# Use Quant Lab's locked Python: python quant/prepare.py --help
"""Fixed Quant evaluator for the fork's editable train.py; never submits or trades."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

B0: Final = "20261028T162312-3f208233"
SETTINGS: Final = {
    "book": "long_only", "delay": 1, "universe": "TOP500",
    "neutralization": "market", "truncation": 0.05, "decay": 0,
    "start": "2015-01-01", "end": "2025-09-11", "cost_bps": 5.0,
    "tax": True, "booksize": 1.0,
}


class EvaluationError(RuntimeError):
    """The fixed study contract or an evaluator invocation failed."""


class OutputLocationError(EvaluationError):
    """Assessment artifacts must remain outside the immutable saved-run home."""


@dataclass(frozen=True, slots=True)
class Study:
    repo: Path
    home: Path
    protocol: Path

    @classmethod
    def from_environment(cls) -> Study:
        return cls(*(Path(os.environ[key]).resolve(strict=True) for key in (
            "QUANTLAB_REPO", "QUANTLAB_HOME", "QUANTLAB_PROTOCOL",
        )))


@dataclass(frozen=True, slots=True)
class Outcome:
    source_sha256: str
    status: str
    run_id: str | None
    net_sharpe: float | None
    annual_net_return: float | None
    promotion_eligible: bool = False
    validity_blocker: str = "unresolved engine future-bar dependency; final tests unperformed"


def validate_candidate(source: str) -> None:
    """Reject changed settings before qlab can read any data."""
    module = ast.parse(source)
    assignments = [node for node in module.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "SETTINGS" for t in node.targets)]
    if len(assignments) != 1 or ast.literal_eval(assignments[0].value) != SETTINGS:
        raise EvaluationError("Candidate must retain the complete fixed Quant SETTINGS")
    if not any(isinstance(node, ast.FunctionDef) and node.name == "alpha" for node in module.body):
        raise EvaluationError("Candidate must define alpha(d)")


def fixed_fingerprint(study: Study) -> str:
    """Detect changes to the trusted engine, evaluator and protocol during a probe."""
    paths = [*sorted((study.repo / "quantlab").rglob("*.py")),
             *sorted((study.repo / "scripts").glob("*.py")),
             study.repo / "scripts/assessment_policy.json", study.repo / "uv.lock",
             study.protocol, Path(__file__)]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def evaluate(candidate: Path, output: Path, hypothesis: str) -> Outcome:
    """Run the original qlab attempt and strict assessment, preserving all evidence."""
    source = candidate.read_text(encoding="utf-8")
    validate_candidate(source)
    study = Study.from_environment()
    if output.resolve().is_relative_to(study.home):
        raise OutputLocationError("--output must be outside QUANTLAB_HOME")
    if not (study.home / "runs" / B0 / "run.json").is_file():
        raise EvaluationError("Study lacks the inherited B0 run; do not reset trial history")
    output.mkdir(parents=True, exist_ok=False)
    (output / "hypothesis.txt").write_text(hypothesis, encoding="utf-8")
    (output / "train.py").write_text(source, encoding="utf-8")
    before = fixed_fingerprint(study)
    command = [sys.executable, str(study.repo / "scripts/autoresearch.py"), "attempt",
               str(output / "train.py"), "--repo", str(study.repo), "--home", str(study.home),
               "--protocol", str(study.protocol), "--hypothesis", hypothesis, "--timeout", "1200"]
    with (output / "attempt.json").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, text=True,
                                 cwd=study.repo, timeout=1260, check=False)
    attempt = json.loads((output / "attempt.json").read_text())
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    result = Outcome(source_hash, "crash", None, None, None)
    if process.returncode == 0 and attempt.get("status") == "completed":
        run_id = attempt["run_id"]
        report = output / "assessment.json"
        subprocess.run([sys.executable, str(study.repo / "scripts/assess_candidate.py"),
                        "--home", str(study.home), "--candidate", run_id, "--benchmark", B0,
                        "--out", str(report)], cwd=study.repo, timeout=120, check=True)
        assessment = json.loads(report.read_text())
        metrics = assessment["runs"]["candidate"]["net"]
        sharpe, annual = metrics["sharpe"], metrics["returns"]
        if sharpe is not None and not math.isfinite(sharpe):
            raise EvaluationError("Nonfinite Sharpe cannot select a candidate")
        result = Outcome(source_hash, "keep" if assessment["passed"] else "discard",
                         run_id, sharpe, annual)
    after = fixed_fingerprint(study)
    (output / "fixed-inputs.json").write_text(json.dumps({"before": before, "after": after}))
    if before != after:
        raise EvaluationError("Fixed engine, evaluator or protocol changed during the probe")
    (output / "result.json").write_text(json.dumps(asdict(result), indent=2, allow_nan=False))
    ledger = study.home / "results.tsv"
    with ledger.open("a", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        if stream.tell() == 0:
            writer.writerow(("source_sha256", "net_sharpe", "status", "description"))
        writer.writerow((source_hash, result.net_sharpe, result.status, hypothesis))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.candidate.resolve(strict=True), args.output.resolve(), args.hypothesis)
    except (EvaluationError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "crash", "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), allow_nan=False))
    return int(result.status == "crash")


if __name__ == "__main__":
    raise SystemExit(main())
