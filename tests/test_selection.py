"""Selection records written by the export cells (backtests/<id>.meta.json)."""
import json

import numpy as np
import pandas as pd

from pcon.backtests import load_meta
from pcon.backtests import tests_summary as summarize_tests
from pcon.book import Book


def _with_backtest(ws, meta=None, start="2020-01-01"):
    (ws / "backtests").mkdir(exist_ok=True)
    idx = pd.bdate_range(start, "2025-12-31")
    pd.DataFrame({"date": idx.strftime("%Y-%m-%d"),
                  "return": np.random.default_rng(0).normal(0.0004, 0.01, len(idx))}).to_csv(
        ws / "backtests" / "a.csv", index=False)
    text = (ws / "portfolio.yaml").read_text().replace("a: {name: A, target_weight: 0.5}",
                                                       "a: {name: A, target_weight: 0.5, backtest: backtests/a.csv}")
    (ws / "portfolio.yaml").write_text(text)
    if meta is not None:
        (ws / "backtests" / "a.meta.json").write_text(json.dumps(meta))


def test_missing_record_is_flagged(workspace):
    _with_backtest(workspace)
    msgs = [h["msg"] for h in Book(workspace, end="2026-01-16").health() if h["scope"] == "A"]
    assert any("no selection record" in m for m in msgs)


def test_failed_final_test_and_override_raise_amber(workspace):
    meta = {"selected": "Rule X", "passed_selection": False,
            "final_tests": {"t1": True, "t2": False, "t3": None}, "oos_start": "2023-01-02"}
    _with_backtest(workspace, meta)
    assert summarize_tests(load_meta(workspace / "backtests" / "a.csv")) == (1, 2, ["t2"])
    b = Book(workspace, end="2026-01-16")
    amber = [h["msg"] for h in b.health() if h["scope"] == "A" and h["level"] == "amber"]
    assert any("NOT the notebook's final selection" in m for m in amber)
    assert any("passed 1/2 final tests; failed: t2" in m for m in amber)
    # no expectation_start in portfolio.yaml -> the record's out-of-sample start is used
    assert b.expectation_start("a") == pd.Timestamp("2023-01-02")
    assert b.expected("a").index[0] >= pd.Timestamp("2023-01-02")


def test_demo_records_present(tmp_path):
    from pcon.demo import build_demo
    b = Book(build_demo(tmp_path / "demo"))
    assert set(b.backtest_meta) == set(b.strategy_ids)
