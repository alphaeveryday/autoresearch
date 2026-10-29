"""Synthetic causal checks must reject future-dependent candidates."""
import os
import tempfile
import unittest
from pathlib import Path

from quant.preflight import PreflightError, check, validate_surface
from quant.prepare import SETTINGS


class PreflightTests(unittest.TestCase):
    def test_unsafe_imports_are_rejected_before_execution(self):
        for body in ("import os", "from pathlib import Path", "open('hidden')"):
            source = f"SETTINGS = {SETTINGS!r}\n{body}\ndef alpha(d):\n    return d.returns\n"
            with self.subTest(body=body), self.assertRaises(PreflightError):
                validate_surface(source)

    @unittest.skipUnless("QUANTLAB_REPO" in os.environ, "requires the locked Quant Lab dependency environment")
    def test_prefix_oracle_rejects_future_returns_but_accepts_causal_signal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.py"
            path.write_text(f"SETTINGS = {SETTINGS!r}\ndef alpha(d):\n    return -d.returns.rolling(5).mean()\n")
            check(path)
            path.write_text(f"SETTINGS = {SETTINGS!r}\ndef alpha(d):\n    return d.returns.shift(-1)\n")
            with self.assertRaises(AssertionError):
                check(path)


if __name__ == "__main__":
    unittest.main()
