"""Intent tests for the fixed Quant profile; no market data is loaded."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quant.prepare import (
    SETTINGS,
    EvaluationError,
    OutputLocationError,
    evaluate,
    validate_candidate,
)


class QuantPolicyTests(unittest.TestCase):
    def source(self, settings):
        return f"SETTINGS = {settings!r}\ndef alpha(d):\n    return d.returns\n"

    def test_fixed_candidate_is_accepted(self):
        # Given exact trading assumptions; when checked; then no rejection.
        validate_candidate(self.source(SETTINGS))

    def test_cheaper_cost_same_close_and_holdout_end_are_rejected(self):
        for key, value in (("cost_bps", 0), ("delay", 0), ("end", "2026-01-01")):
            with self.subTest(key=key):
                # Given a candidate changing a fixed assumption.
                changed = SETTINGS | {key: value}
                # When checked, then reject before any evaluator invocation.
                with self.assertRaises(EvaluationError):
                    validate_candidate(self.source(changed))

    def test_missing_alpha_and_duplicate_settings_are_rejected(self):
        for source in (f"SETTINGS = {SETTINGS!r}", self.source(SETTINGS) + f"SETTINGS = {SETTINGS!r}"):
            # Given ambiguous/missing contract; when checked; then reject.
            with self.subTest(source=source), self.assertRaises(EvaluationError):
                validate_candidate(source)

    def test_output_inside_study_is_rejected_before_a_run(self):
        # Given an output path nested in the saved-run home.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "train.py"
            candidate.write_text(self.source(SETTINGS))
            protocol = root / "protocol.md"
            protocol.write_text("fixed")
            environment = {"QUANTLAB_REPO": str(root), "QUANTLAB_HOME": str(root),
                           "QUANTLAB_PROTOCOL": str(protocol)}
            # When evaluated, then fail on location before reading a B0 or launching qlab.
            with patch.dict(os.environ, environment), self.assertRaises(OutputLocationError):
                evaluate(candidate, root / "output", "test")


if __name__ == "__main__":
    unittest.main()
