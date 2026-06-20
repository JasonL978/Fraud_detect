"""Day 2 demo: map one PaySim row to transactions.v1 and validate it.

Run: python scripts/map_demo.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detect.paysim_mapping import map_paysim_row, validate_record  # noqa: E402

# A real PaySim fraud row: a TRANSFER that drains the sender's account.
ROW = {
    "step": 1, "type": "TRANSFER", "amount": 181.0,
    "nameOrig": "C1231006815", "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
    "nameDest": "C1666544295", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    "isFraud": 1, "isFlaggedFraud": 0,
}


def _jsonable(rec):
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in rec.items()}


if __name__ == "__main__":
    rec = map_paysim_row(ROW, row_index=0)
    print("valid against schema:", validate_record(rec))
    print(json.dumps(_jsonable(rec), indent=2))
