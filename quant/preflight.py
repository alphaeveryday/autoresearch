# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# ─── How to run ───
# QUANTLAB_REPO=/locked/repo /locked/python -m quant.preflight /absolute/candidate.py
"""Trusted synthetic checks before any candidate receives historical prices."""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
from pathlib import Path

from quant.prepare import validate_candidate


class PreflightError(RuntimeError):
    """A candidate violates the causal research contract."""


def validate_surface(source: str) -> None:
    """Restrict candidate imports and obvious I/O; this is not a security sandbox."""
    validate_candidate(source)
    blocked = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "globals", "locals"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            raise PreflightError("Use only imports from quantlab.alpha.ops")
        if isinstance(node, ast.ImportFrom) and node.module != "quantlab.alpha.ops":
            raise PreflightError("Use only imports from quantlab.alpha.ops")
        if isinstance(node, ast.Name) and node.id in blocked:
            raise PreflightError("Dynamic execution or file access is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith(("_", "read_", "to_csv", "to_parquet", "to_pickle")):
            raise PreflightError("Private attributes and file I/O are not allowed")


def check(path: Path) -> None:
    validate_surface(path.read_text())
    sys.path.insert(0, os.environ["QUANTLAB_REPO"])
    from quantlab.alpha.loader import load
    from quantlab.alpha.ops import np, pd
    from quantlab.data.panel import Data

    rng = np.random.default_rng(20261030)
    dates = pd.bdate_range("2018-01-01", periods=900)
    codes = pd.Index([f"{i:05d}0" for i in range(40)])
    returns = rng.normal(0.0001, 0.02, (900, 40))
    close = 10000 * np.cumprod(1 + returns, axis=0)
    volume = rng.lognormal(12, 1, (900, 40))
    fields = {"close": close, "open": close / (1 + returns), "high": close * 1.02,
              "low": close * .98, "returns": returns, "volume": volume,
              "turnover": close * volume, "cap": close * np.arange(1, 41)[None, :] * 1e6,
              "shares": np.ones((900, 40)) * 1e6, "vwap": close}
    frames = {name: pd.DataFrame(values, index=dates, columns=codes) for name, values in fields.items()}
    for frame in frames.values():
        frame.iloc[:120, :4] = np.nan
        frame.iloc[600:, 4:6] = np.nan
    frames["volume"].iloc[500:520, 6:8] = 0
    with tempfile.TemporaryDirectory() as temporary:
        # Data is constructed by its normal API, then synthetic frames are installed.
        import duckdb
        rows = pd.DataFrame({"date": np.repeat(dates, len(codes)), "code": np.tile(codes, len(dates))})
        source = Path(temporary) / "synthetic.parquet"
        duckdb.from_df(rows).write_parquet(str(source))
        data = Data("2018-01-01", "2021-12-31", str(source), warmup_days=0)
        data._frames = frames
        data.dates, data.codes = dates, codes
        candidate = load(path)
        full = candidate.fn(data)
        if not full.index.equals(dates) or not full.columns.equals(codes):
            raise PreflightError("Output axes differ from the supplied panel")
        if np.isinf(full.to_numpy()).any() or np.nanmax(np.abs(full.to_numpy())) == 0:
            raise PreflightError("Infinite or entirely inactive synthetic signal")
        repeated = candidate.fn(data)
        np.testing.assert_allclose(full, repeated, rtol=1e-9, atol=1e-11, equal_nan=True)
        for cutoff in (499, 699, 849):
            prefix = candidate.fn(data.truncated(dates[cutoff]))
            np.testing.assert_allclose(full.iloc[:cutoff + 1], prefix, rtol=1e-7, atol=1e-9, equal_nan=True)
        changed = Data("2018-01-01", "2021-12-31", str(source), warmup_days=0)
        changed.dates, changed.codes = dates, codes
        changed._frames = {name: frame.copy() for name, frame in frames.items()}
        for frame in changed._frames.values():
            frame.iloc[700:] = frame.iloc[700:] * 1.7
        future = candidate.fn(changed)
        np.testing.assert_allclose(full.iloc[:700], future.iloc[:700], rtol=1e-7, atol=1e-9, equal_nan=True)


if __name__ == "__main__":
    check(Path(sys.argv[1]))
    print(json.dumps({"passed": True, "synthetic_days": 900, "codes": 40,
                      "prefix_cutoffs": [499, 699, 849], "future_mutation_after": 699,
                      "repeatability": True, "market_data_read": False}))
