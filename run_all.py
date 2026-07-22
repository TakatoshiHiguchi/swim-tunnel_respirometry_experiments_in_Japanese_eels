from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def run(cmd: list[str]) -> None:
    print("\n>>> " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    py = sys.executable
    run([py, str(SCRIPTS / "01_validate_inputs.py")])
    run([py, str(SCRIPTS / "02_reanalysis_individual_endpoints.py")])
    run([py, str(SCRIPTS / "03_speed_step_summaries.py")])
    run([py, str(SCRIPTS / "04_make_figures.py")])
    print("\nAll analyses completed.")


if __name__ == "__main__":
    main()
