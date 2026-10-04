"""Run the three strategies live: execute each research notebook headless on fresh data, then its export cell.

    python -m pcon update                 # all strategies, one after the other (what the dashboard button runs)
    python -m pcon update --only regime_filter

The notebooks in ``strategies/`` are the validated research code, unchanged: this module only runs their code
cells (no Jupyter needed), switches their data refresh on, points their caches at ``<workspace>/cache`` and appends
the matching ``strategy_exports/`` cell, which writes ``backtests/<id>.csv`` (+ the selection record) and
``signals/<id>.json`` into the workspace. Each strategy runs in its own Python process; progress goes to
``<workspace>/cache/update/``.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import traceback
import types
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HOUR = 3600.0
DAY = 24 * HOUR


@dataclass
class Spec:
    notebook: str
    export: str
    subs: list[tuple[str, str]] = field(default_factory=list)      # exact text replacements in the code cells
    expire: list[tuple[str, float]] = field(default_factory=list)   # (glob under the work dir, max age in seconds)
    minutes: str = ""                                               # typical run time, shown in the dashboard


SPECS: dict[str, Spec] = {
    "rolling_momentum": Spec(
        "strategies/rolling_momentum/Rolling_Momentum_Report.ipynb", "strategy_exports/export_rolling_momentum.py",
        subs=[('"refresh_data": False,', '"refresh_data": True,')], minutes="under a minute"),
    "regime_filter": Spec(
        "strategies/regime_filter/regime_filter_backtest.ipynb", "strategy_exports/export_regime_filter.py",
        subs=[("REFRESH_DATA = False", "REFRESH_DATA = True")], minutes="a few minutes"),
    "sma_piotroski": Spec(
        "strategies/sma_piotroski/SMA_Piotroski_Backtest.ipynb", "strategy_exports/export_sma_piotroski.py",
        # REFRESH_DATA re-downloads everything; instead stale cache files are deleted so only they are fetched again
        expire=[("data_cache/prices/*", 12 * HOUR), ("data_cache/sp500_*.csv", 7 * DAY),
                ("data_cache/sec_company_tickers.json", 7 * DAY), ("data_cache/sec/*", 7 * DAY)],
        minutes="5-15 minutes (the first run downloads ~900 price histories and SEC filings: 20-40 minutes)"),
}


# ---- settings ------------------------------------------------------------------------------------
def settings_path(ws: Path) -> Path:
    return Path(ws) / "settings.json"


def load_settings(ws: Path) -> dict:
    p = settings_path(ws)
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def save_settings(ws: Path, **kw) -> dict:
    s = load_settings(ws) | {k: v for k, v in kw.items() if v is not None}
    settings_path(ws).write_text(json.dumps(s, indent=2))
    return s


def valid_sec_contact(s: str) -> bool:
    """The SEC wants 'Name email@domain' in the User-Agent."""
    return bool(re.search(r"\S+\s+\S+@\S+\.\S+", s or "")) and "example.com" not in s


# ---- status --------------------------------------------------------------------------------------
def update_dir(ws: Path) -> Path:
    d = Path(ws) / "cache" / "update"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_status(ws: Path) -> dict:
    p = update_dir(ws) / "status.json"
    try:
        st = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        st = {}
    if st.get("running") and not _alive(st.get("pid")):
        st["running"], st["crashed"] = None, True          # the updater died (reboot, closed window)
    return st


def _write_status(ws: Path, st: dict) -> None:
    p = update_dir(ws) / "status.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, default=str))
    tmp.replace(p)


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}"], capture_output=True, text=True).stdout
            return str(int(pid)) in out
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def is_running(ws: Path) -> bool:
    return bool(read_status(ws).get("running"))


def log_tail(ws: Path, sid: str, n: int = 12) -> str:
    p = update_dir(ws) / f"{sid}.log"
    if not p.exists():
        return ""
    lines = [l for l in p.read_text(errors="replace").splitlines() if l.strip()]
    return "\n".join(lines[-n:])


# ---- orchestration (parent process) --------------------------------------------------------------
def start_update(ws: Path, only: list[str] | None = None) -> bool:
    """Start ``python -m pcon update`` in the background (survives dashboard reruns). False if one is running."""
    ws = Path(ws).resolve()
    if is_running(ws):
        return False
    cmd = [sys.executable, "-m", "pcon", "update", "--workspace", str(ws)] + (["--only", *only] if only else [])
    kw = dict(cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kw["start_new_session"] = True
    p = subprocess.Popen(cmd, **kw)
    _write_status(ws, {"running": "starting", "pid": p.pid, "started": _now(), "results": read_status(ws).get("results", {})})
    return True


def update_all(ws: Path, only: list[str] | None = None) -> dict:
    """Run each strategy in its own process, one after the other, recording progress in status.json."""
    from .config import load_config
    ws = Path(ws).resolve()
    cfg = load_config(ws)
    sids = [s for s in (only or SPECS) if s in SPECS and s in cfg.strategies]
    st = read_status(ws)
    st.update(running=None, pid=os.getpid(), started=_now(), finished=None, crashed=False)
    st.setdefault("results", {})
    for sid in sids:
        st["running"] = sid
        _write_status(ws, st)
        t0 = time.time()
        log = update_dir(ws) / f"{sid}.log"
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run([sys.executable, "-u", "-m", "pcon.runner", sid, str(ws)], cwd=str(ROOT),
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        tail = log_tail(ws, sid, 40).splitlines()
        msg = next((l for l in reversed(tail) if l.startswith(("ERROR", "Signal", "Next open"))), tail[-1] if tail else "")
        st["results"][sid] = {"ok": rc == 0, "finished": _now(), "seconds": round(time.time() - t0),
                              "message": msg[:400]}
        _write_status(ws, st)
        print(f"{sid}: {'ok' if rc == 0 else 'FAILED'} in {time.time() - t0:.0f}s  {msg}", flush=True)
    st.update(running=None, finished=_now())
    _write_status(ws, st)
    return st


def _now() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


# ---- one notebook (child process) ----------------------------------------------------------------
def _stub_ipython() -> None:
    """The notebooks only use IPython.display for display()/Markdown(): silence it, or provide it when IPython is
    not installed (no Jupyter needed headless)."""
    noop = lambda *a, **k: None
    try:
        import IPython.display as disp
        disp.display = noop
        return
    except ImportError:
        pass
    disp = types.ModuleType("IPython.display")
    disp.display = noop
    disp.Markdown = disp.HTML = disp.Image = lambda *a, **k: (a[0] if a else None)
    ip = types.ModuleType("IPython")
    ip.display, ip.get_ipython = disp, noop
    sys.modules["IPython"], sys.modules["IPython.display"] = ip, disp


def _expire(work: Path, rules: list[tuple[str, float]]) -> int:
    now, n = time.time(), 0
    for pattern, age in rules:
        for f in work.glob(pattern):
            if f.is_file() and now - f.stat().st_mtime > age:
                f.unlink()
                n += 1
    return n


def _sma_live_start(ws: Path) -> str | None:
    """First day you traded the SMA sleeve: the export then follows a model account opened that day."""
    p = Path(ws) / "trades.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    t = t[t.get("strategy", pd.Series(dtype=str)) == "sma_piotroski"]
    return str(pd.to_datetime(t["date"]).min().date()) if len(t) else None


def code_cells(nb_path: Path, subs: list[tuple[str, str]]) -> list[str]:
    cells = []
    for c in json.loads(nb_path.read_text(encoding="utf-8"))["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        for a, b in subs:
            src = src.replace(a, b)
        cells.append(src)
    return cells


def run_one(sid: str, ws: Path) -> None:
    ws = Path(ws).resolve()
    spec = SPECS[sid]
    nb = ROOT / spec.notebook
    work = ws / "cache" / "strategies" / sid
    work.mkdir(parents=True, exist_ok=True)
    settings = load_settings(ws)
    env = {"PCON_BACKTESTS": str(ws / "backtests"), "PCON_SIGNALS": str(ws / "signals"), "MPLBACKEND": "Agg",
           "SMA_PF_CACHE": str(work / "data_cache"), "SMA_PF_REPORTS": str(work / "reports")}
    if settings.get("sec_contact"):
        env["SEC_USER_AGENT"] = settings["sec_contact"]
    if settings.get("fred_api_key"):
        env["FRED_API_KEY"] = settings["fred_api_key"]
    os.environ.update(env)
    if sid == "sma_piotroski" and not valid_sec_contact(os.environ.get("SEC_USER_AGENT", "")):
        print("ERROR: the SEC requires a contact for its filings API. Enter your name and e-mail under "
              "'Data sources' in the Orders tab, then update again.", flush=True)
        sys.exit(2)
    os.chdir(work)
    sys.path.insert(0, str(nb.parent))
    _stub_ipython()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: plt.close("all")
    if spec.expire:
        print(f"Expired {_expire(work, spec.expire)} stale cache files", flush=True)
    export = (ROOT / spec.export).read_text(encoding="utf-8")
    if sid == "sma_piotroski" and (live := _sma_live_start(ws)):
        export = export.replace("LIVE_START = None", f'LIVE_START = "{live}"')
    cells = code_cells(nb, spec.subs)
    g: dict = {"__name__": "__main__"}
    t0 = time.time()
    for k, src in enumerate(cells, 1):
        print(f"@@ cell {k}/{len(cells)}", flush=True)
        try:
            exec(compile(src, f"{nb.name} cell {k}", "exec"), g)
        except SystemExit:
            raise
        except Exception as exc:
            traceback.print_exc()
            print(f"ERROR in notebook cell {k}: {type(exc).__name__}: {exc}", flush=True)
            sys.exit(1)
        plt.close("all")
    print(f"@@ notebook done in {time.time() - t0:.0f}s; exporting", flush=True)
    try:
        exec(compile(export, spec.export, "exec"), g)
    except Exception as exc:
        traceback.print_exc()
        print(f"ERROR in the export cell: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)


if __name__ == "__main__":          # python -m pcon.runner <strategy> <workspace>
    run_one(sys.argv[1], Path(sys.argv[2]))
