# Infra — Local Dev Stack

Week 1 Day 1 stands up the data plane: **Redpanda** (Kafka API + built-in
Schema Registry) and the **Redpanda Console** UI, with the `transactions.v1`
topic created automatically on startup.

> Redpanda bundles the Schema Registry into the same binary, so there is no
> separate Confluent registry container. The registry is reachable on
> `http://localhost:18081`.

## Prerequisites

- Docker Desktop running (Compose v2 — `docker compose`, not `docker-compose`).

### Installing from zero on Windows 11

Docker Desktop is **free for personal / portfolio use** (paid only for orgs
with 250+ employees or $10M+ revenue). It runs on the WSL2 backend, so WSL2
must be installed first.

Run these in an **Administrator** PowerShell:

```powershell
# 1. Install WSL2, then REBOOT when prompted.
wsl --install
#    (fallback if that errors: `wsl --install --no-distribution` then `wsl --install -d Ubuntu`)

# 2. After reboot, install Docker Desktop.
winget install --id Docker.DockerDesktop -e --accept-source-agreements --accept-package-agreements
```

Then launch **Docker Desktop** from the Start menu (first start initializes the
WSL2 backend — give it a minute) and verify in a normal terminal:

```powershell
docker --version
docker compose version
```

## Bring it up

```bash
cp .env.example .env          # first time only
docker compose up -d
```

The `topic-init` one-shot service waits for the broker to report healthy,
creates `transactions.v1`, prints the topic list, then exits 0. Re-running
`up` is safe — topic creation is idempotent.

## Verify the Day 1 "done" criteria

1. **Topic exists:**
   ```bash
   docker compose exec redpanda rpk topic list
   ```
   Expect `transactions.v1` with 3 partitions.

2. **Schema Registry reachable:**
   ```bash
   curl -s http://localhost:18081/subjects
   ```
   Expect `[]` (empty list — no schemas registered until Day 2). A `200`
   with `[]` means the registry is up.

3. **Console UI:** open http://localhost:8080 — the topic should be listed.

## Ports

| Port  | Service                         | Used by              |
|-------|---------------------------------|----------------------|
| 19092 | Kafka API (external listener)   | host producers/scripts |
| 9092  | Kafka API (internal listener)   | other containers     |
| 18081 | Schema Registry                 | host                 |
| 18082 | HTTP proxy (pandaproxy)         | host                 |
| 9644  | Admin API                       | tooling/health       |
| 8080  | Redpanda Console                | browser              |

The **external** listener (`localhost:19092`) is for clients on your machine;
the **internal** listener (`redpanda:9092`) is for other containers. A producer
run from your laptop must use `localhost:19092`.

## Tear down

```bash
docker compose down        # stop, keep data
docker compose down -v     # stop and wipe the redpanda-data volume
```

## Not in Day 1 (deliberately deferred)

- **Avro schema for `transactions.v1`** — Day 2 (target schema design).
- **Redis / MinIO** — not needed until online features (Week 2) / offline
  store. Kept out of the compose file to keep Day 1 minimal.
- **AUTH/TLS on listeners** — documented as deferred config; this is a local
  single-host dev stack with no exposed ports beyond localhost.
