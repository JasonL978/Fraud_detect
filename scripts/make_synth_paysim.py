"""Write a synthetic PaySim-like CSV for offline pipeline validation.

Run: python scripts/make_synth_paysim.py --rows 20000
Output: data/raw/paysim_synth_demo.csv  (gitignored)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detect.synth import make_synthetic_paysim  # noqa: E402

_OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "paysim_synth_demo.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    df = make_synthetic_paysim(args.rows, args.seed)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUT, index=False)
    print(f"wrote {len(df):,} rows to {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
