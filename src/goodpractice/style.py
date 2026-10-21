"""Plot style: colour-blind-safe categorical palette, light background."""
from pathlib import Path

import matplotlib as mpl

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY = "#8a8985"
GRID = "#e4e3df"


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=CAT), "legend.frameon": False,
        "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig, name: str, folder: Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{name}.png"
    fig.savefig(out)
    return out
