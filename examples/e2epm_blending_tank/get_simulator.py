"""Download the open E2EPM blending-tank simulator (MIT licence) into ./external.

    python get_simulator.py

Clones https://github.com/LidiaAuret/ADCHEM2024_End-to-end-process-monitoring-workshop
at a pinned commit, so results are reproducible. Requires git.
"""

import subprocess
from pathlib import Path

URL = "https://github.com/LidiaAuret/ADCHEM2024_End-to-end-process-monitoring-workshop.git"
COMMIT = "792a64f8af779a623c78bbffa7504ede841a2b33"  # 2025-07-04
DEST = Path(__file__).resolve().parent / "external" / "ADCHEM2024_E2EPM"


def main() -> None:
    if DEST.exists():
        print(f"{DEST} exists; nothing to do")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", URL, str(DEST)], check=True)
    subprocess.run(["git", "-C", str(DEST), "checkout", COMMIT], check=True)
    print(f"simulator ready in {DEST}")


if __name__ == "__main__":
    main()
