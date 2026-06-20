"""Day 4 demo: RT-014 schema validator routes unknown enums to a review bucket.

Run: python scripts/validate_demo.py

Shows a good record accepted and two malformed records routed to review with
reason codes -- crucially, nothing crashes and no unknown value is silently
encoded as all-zeros.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detect.paysim_mapping import map_paysim_row  # noqa: E402
from fraud_detect.schema_validator import ReviewRouter  # noqa: E402

_ROW = {
    "step": 1, "type": "TRANSFER", "amount": 181.0,
    "nameOrig": "C1231006815", "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
    "nameDest": "C1666544295", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    "isFraud": 1, "isFlaggedFraud": 0,
}


def main() -> int:
    good = map_paysim_row(_ROW, row_index=0)

    # Two events that bypassed the trusted mapper (e.g. via the JSON proxy path):
    bogus_type = dict(good, tx_type="CRYPTO")          # unknown tx_type
    bogus_channel = dict(good, channel="CARRIER_PIGEON")  # unknown nullable enum

    accepted, review = [], []
    router = ReviewRouter(
        on_accept=lambda r: accepted.append(r["event_id"]),
        on_review=lambda r, reasons: review.append((r.get("event_id"), reasons)),
    )

    for rec in (good, bogus_type, bogus_channel):
        result = router.route(rec)
        print(f"{rec.get('tx_type'):<8} channel={rec.get('channel')!r:<18} -> {result.status} {result.reasons}")

    print(f"\naccepted={len(accepted)} routed_to_review={len(review)}")
    assert len(accepted) == 1 and len(review) == 2, "demo invariant failed"
    print("OK: unknown enums diverted to review, nothing zero-encoded or crashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
