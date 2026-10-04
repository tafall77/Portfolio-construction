"""Create a new workspace (folder with portfolio.yaml, journal CSVs and backtests/)."""
from __future__ import annotations

import shutil
from pathlib import Path

from .journal import init_files

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "workspace"


def init_workspace(root: str | Path) -> Path:
    """Copy the template into ``root`` without overwriting anything that already exists."""
    root = Path(root)
    (root / "backtests").mkdir(parents=True, exist_ok=True)
    if not (root / "portfolio.yaml").exists():
        shutil.copy(TEMPLATE / "portfolio.yaml", root / "portfolio.yaml")
    init_files(root)
    readme = root / "backtests" / "README.md"
    if not readme.exists():
        readme.write_text("Put each strategy's exported daily returns here (see strategy_exports/ in the repo):\n"
                          "sma_piotroski.csv, regime_filter.csv, rolling_momentum.csv\n")
    return root
