"""The live runner: notebooks executed headless on fresh data, then their export cell."""
import json
import os
import time

import pytest

from pcon import runner as RUN


def _notebook(path, *cells):
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": c, "metadata": {}, "outputs": []}
                                          for c in cells] + [{"cell_type": "markdown", "source": "x"}]}))


def test_code_cells_drop_magics_and_apply_switches(tmp_path):
    nb = tmp_path / "n.ipynb"
    _notebook(nb, "%matplotlib inline\nREFRESH = False\n!pip install x\nprint(1)")
    assert RUN.code_cells(nb, [("REFRESH = False", "REFRESH = True")]) == ["REFRESH = True\nprint(1)"]


def test_sec_contact_and_settings(tmp_path):
    assert RUN.valid_sec_contact("Jane Doe jane@mail.org")
    assert not RUN.valid_sec_contact("jane@mail.org") and not RUN.valid_sec_contact("Your Name your.email@example.com")
    RUN.save_settings(tmp_path, sec_contact="Jane Doe jane@mail.org")
    RUN.save_settings(tmp_path, fred_api_key="k")
    assert RUN.load_settings(tmp_path) == {"sec_contact": "Jane Doe jane@mail.org", "fred_api_key": "k"}


def test_expire_only_removes_stale_files(tmp_path):
    (tmp_path / "prices").mkdir()
    old, new = tmp_path / "prices" / "A.csv.gz", tmp_path / "prices" / "B.csv.gz"
    old.write_text("x"), new.write_text("y")
    os.utime(old, (time.time() - 2 * RUN.DAY, time.time() - 2 * RUN.DAY))
    assert RUN._expire(tmp_path, [("prices/*", RUN.DAY)]) == 1
    assert not old.exists() and new.exists()


def test_update_runs_notebook_then_export(workspace, tmp_path, monkeypatch):
    nb, ex = tmp_path / "toy.ipynb", tmp_path / "export_toy.py"
    _notebook(nb, "import os\nSIGNAL = 'SPY' if FLAG else 'cash'", "WHERE = os.getcwd()")
    ex.write_text("import json, os\nfrom pathlib import Path\n"
                  "d = Path(os.environ['PCON_SIGNALS']); d.mkdir(exist_ok=True)\n"
                  "(d / 'a.json').write_text(json.dumps({'kind': 'weights', 'as_of': '2026-01-16', "
                  "'weights': {SIGNAL: 1.0}, 'cwd': WHERE}))\nprint('Signal: done')\n")
    monkeypatch.setitem(RUN.SPECS, "a", RUN.Spec(str(nb), str(ex), subs=[("FLAG", "True")]))
    if not hasattr(os, "fork"):
        pytest.skip("needs fork")
    pid = os.fork()                               # run_one changes cwd and sys.modules: keep it out of pytest
    if pid == 0:
        try:
            RUN.run_one("a", workspace)
        finally:
            os._exit(0)
    os.waitpid(pid, 0)
    sig = json.loads((workspace / "signals" / "a.json").read_text())
    assert sig["weights"] == {"SPY": 1.0}
    assert sig["cwd"].endswith(os.path.join("cache", "strategies", "a"))
