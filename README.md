# Real-Time P2P Payment Fraud Detection

> **Status:** Week 1, Day 1 — data plane skeleton.
>
> **All fraud metrics in this project are synthetic-on-synthetic.** The base
> data is PaySim (a generated mobile-money dataset); APP-fraud, device, and
> graph signals are injected by code in this repo for *training*, and the
> evaluation set is produced by a *separate* generator. The numbers demonstrate
> **systems behavior**, not detection performance on real fraud. This caveat is
> load-bearing — see the design doc.

A streaming + serving + retraining fraud-detection system for a P2P payments
archetype (Venmo / Cash App / Zelle), built to a hardened, threat-modeled
design. See `p2p-fraud-detection-design-doc-v1.1.md` for the full design and
the descoped v1.2 build plan.

## The spine (the part that matters)

1. **Online/offline feature parity test** — guards against train/serve skew.
2. **Decision Shim** — boundary layer that strips score/reason and rate-limits,
   defeating the oracle attack class.
3. **Canary guardrail** — promotion gate that blocks poisoned/regressed retrains.
4. **Threat-model writeup** — including the honest gap between training-set and
   independent-eval recall.

Everything else is a labeled stretch goal.

## Quick start (Day 1)

```bash
cp .env.example .env
docker compose up -d
docker compose exec redpanda rpk topic list      # expect transactions.v1
curl -s http://localhost:18081/subjects          # expect [] (registry up)
```

See [infra/README.md](infra/README.md) for details and the verify checklist.

## Repo layout (grows over the build)

```
docker-compose.yml      # local dev stack (Redpanda + Console)
infra/                  # infra configs + runbook
data/                   # PaySim + generators (gitignored raw data)   [Day 2+]
features/               # Feast defs + streaming/batch features        [Week 2]
models/                 # baseline, eval, canary set                   [Day 5+]
serving/                # inference / decision_shim / intervention     [Week 3]
```
