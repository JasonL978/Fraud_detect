# Data

Raw data lives in `data/raw/` and is **gitignored** — never committed.

## PaySim (base dataset)

The project's base data is **PaySim**, a synthetic mobile-money transaction
dataset (Lopez-Rojas et al. 2016) — the de facto fraud-detection benchmark
where real data is unavailable.

- **Source:** Kaggle — `ealaxi/paysim1`
- **File:** `PS_20174392719_1491204439457_log.csv` (~470 MB, ~6.3M rows)
- **Place it at:** `data/raw/paysim.csv`

### Download

```bash
# Requires a Kaggle account + API token (~/.kaggle/kaggle.json)
pip install kaggle
kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip
# then rename the extracted CSV to paysim.csv
```

Or download manually from https://www.kaggle.com/datasets/ealaxi/paysim1 and
drop the CSV at `data/raw/paysim.csv`.

### Columns (raw PaySim)

`step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest,
oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`

> **Note:** PaySim's shape is NOT the project's event shape. Day 2 defines the
> target `transactions.v1` Avro schema and maps PaySim → that schema (real
> event timestamp from `step`, sender/recipient IDs, nullable fields for
> device/funding/geo to be injected later). See the design doc.

## Generated data (later)

`data/generators/` (Day 2+) will hold the replay producer, scam injector
(training only), independent eval generator (eval only), and device stream —
all writing synthetic data that is also gitignored.
