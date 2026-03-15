# ZhiSync

ZhiSync is a metadata-driven collaborative inference framework for decentralized healthcare IoT. Edge devices exchange lightweight confidence and urgency metadata (ZhiTag) over UDP so each device can refine its own predictions without sharing raw sensor data or relying on a central server.

This repository contains two layers:

1. **SDK** (`zhisync/`) — the reusable plug-and-play protocol library (no ML dependencies).
2. **Simulation** (`simulation/`) — the full experimental framework from the paper, including neural network models, dataset loaders, training, and the multi-run simulation and scalability harness.

---

---

## Repository Structure

```
zhisync/
  node.py          ZhiSyncNode runtime — per-inference fusion and ZhiTag publishing
  transport.py     UDP JSON transport and in-memory transports; communication stats
  fusion.py        Urgency derivation, staleness filter, peer-context summarization
  types.py         ZhiTag and InferenceDecision dataclasses

simulation/
  models/
    ecg_model.py      ECGNet — 1D CNN for arrhythmia classification
    breath_model.py   CovidCoughNet — Inception + DeepConvNet for cough detection
    motion_model.py   MotionMLP — fully-connected MLP for activity recognition
  datasets/
    ecg_dataset.py    MIT-BIH + PTB loader; synthetic fallback
    breath_dataset.py COUGHVID loader (librosa features); synthetic fallback
    motion_dataset.py WISDM loader with windowed feature extraction; synthetic fallback
  devices/
    base_device.py    BaseDevice implementing ZhiAware Algorithm 1 exactly
    ecg_device.py     ECG wearable monitor (d_ecg)
    breath_device.py  Breath analyzer (d_br)
    motion_device.py  Motion sensor (d_mtn)
  metrics.py          psutil ResourceSampler (1 Hz CPU% + RSS), WallTimer, CommStats
  train.py            Training script — 10 epochs, batch 32, CrossEntropy, Adam
  run_simulation.py   Main experiment — 10 runs × With/Without ZhiSync
  scalability_test.py Scalability sweep — N=4..10 nodes at 2 Hz and 5 Hz

examples/
  integration_minimal.py   In-process two-node demo (in-memory transport)
  udp_node.py              Single UDP node runner

scripts/
  run_udp_demo.sh          Launch 3-node UDP mesh demo
```

---

## Install

SDK only (no ML dependencies):
```bash
python3 -m pip install -e .
```

SDK + simulation framework:
```bash
python3 -m pip install -e ".[simulation]"
# or
pip install -r requirements.txt
```

Requires Python 3.10 or later.

---

## SDK Quick Start

```python
from zhisync.node import NodeOptions, ZhiSyncNode
from zhisync.transport import UdpJsonTransport, UdpPeer

transport = UdpJsonTransport(
    node_id="ECG",
    bind_host="0.0.0.0",
    bind_port=6001,
    peers=[UdpPeer("10.0.0.12", 6003), UdpPeer("10.0.0.13", 6005)],
)

node = ZhiSyncNode(
    options=NodeOptions(
        node_id="ECG",
        urgency_threshold=0.8,   # τ — threshold for HIGH urgency
        staleness_seconds=2.0,   # K — discard peer tags older than this
        metadata_enabled=True,
    ),
    transport=transport,
)

node.start()
try:
    local_conf = my_local_model.predict_confidence(x_t)
    peer_tags  = node.get_fresh_peer_tags()
    context_conf = my_context_model.predict_confidence(x_t, peer_tags)
    decision = node.process(local_confidence=local_conf, context_confidence=context_conf)
    print(decision.final_confidence, decision.urgency, decision.confidence_gain)
finally:
    node.stop()
```

### Communication statistics

`UdpJsonTransport` tracks bytes and packets sent/received, coalesced-suppressed sends, and stale-drop events:

```python
stats = transport.get_stats()
# {'bytes_sent': ..., 'bytes_received': ..., 'packets_sent': ...,
#  'packets_received': ..., 'coalesce_suppressed': ..., 'stale_drops': ...}
```

To enable send coalescing (suppresses redundant broadcasts within a minimum interval):
```python
transport = UdpJsonTransport(..., coalesce_ms=200.0)  # suppress if < 200 ms since last send
```

---

## Local Examples

```bash
# In-process two-node demo (no network required)
python3 examples/integration_minimal.py

# 3-node UDP mesh (ECG, Motion, Breath) — writes logs to runs/
./scripts/run_udp_demo.sh
```

---

## Simulation Framework

### Datasets

The simulation supports real datasets or falls back to synthetic data automatically.

| Modality | Dataset | Source |
|---|---|---|
| ECG | MIT-BIH + PTB Diagnostic | [Kaggle: heartbeat](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) — `mitbih_train.csv`, `ptbdb_normal.csv`, `ptbdb_abnormal.csv` |
| Breath | COUGHVID | [Zenodo 4048312](https://zenodo.org/record/4048312) — audio files + `metadata.csv` |
| Motion | WISDM HAR | [Fordham WISDM](https://www.cis.fordham.edu/wisdm/dataset.php) — `WISDM_ar_v1.1_raw.txt` |

Place downloaded files under a shared data directory, e.g.:
```
/data/zhisync/
  mitbih_train.csv
  mitbih_test.csv
  ptbdb_normal.csv
  ptbdb_abnormal.csv
  coughvid/
    metadata.csv
    *.wav
  WISDM_ar_v1.1_raw.txt
```

If real datasets are absent, synthetic data matching the modality statistics is generated automatically — no download required to run the simulation.

### Training

Trains both the local (baseline) model and the context-aware (ZhiAware) variant for each modality:

```bash
# With synthetic data (no download needed):
python3 -m simulation.train

# With real datasets:
python3 -m simulation.train --data-dir /data/zhisync

# Options:
#   --epochs 10   (default, matches paper)
#   --device cpu/cuda/mps
```

Saved weights go to `simulation/saved_models/` (6 files: `ecg_local.pt`, `ecg_context.pt`, etc.).

### Main Simulation

Runs 10 repeated experiments under two conditions (With ZhiSync / Without ZhiSync), collecting confidence gain, CPU%, RSS, runtime, and communication byte rates:

```bash
python3 -m simulation.run_simulation

# With real datasets and full paper scale:
python3 -m simulation.run_simulation --data-dir /data/zhisync --runs 10 --steps 360

# Quick test:
python3 -m simulation.run_simulation --steps 50 --runs 2

# Options:
#   --runs N       number of repeated experiments (default 10)
#   --steps N      inference steps per device per run (default 360)
#   --sleep-s S    interval between steps, seconds (default 1.0)
#   --output-dir   results saved as JSON (default simulation/results/)
#   --retrain      force model retraining before simulation
```

### Scalability Test

Sweeps N = 4 to 10 nodes at 2 Hz and 5 Hz, measuring per-node byte rate with coalescing:

```bash
python3 -m simulation.scalability_test

# Options:
#   --duration-s 30   seconds per experiment point (default 30)
#   --runs 3          repetitions per point (default 3)
```

---

## ZhiAware Algorithm (Algorithm 1)

Implemented in `simulation/devices/base_device.py`:

1. Run local model → `local_confidence`
2. Fetch fresh peer ZhiTags (within staleness window K)
3. If peers available: build context tensor `[peer_conf, peer_urgency_binary]`
4. Run context-aware model: `ŷ_t = f_θ([x_t; peer_conf, peer_urgency])`
5. Extract confidence `c_t = max(softmax(ŷ_t))`
6. Derive urgency: `u_t = HIGH if c_t ≥ τ else LOW`
7. Broadcast `ZhiTag_t = {device_id, c_t, u_t, timestamp}` to peers

---

## Integration Contract

| Item | Detail |
|---|---|
| Input | `local_confidence` (float 0–1) from your local model; optional `context_confidence` from your context-aware model |
| Output | `InferenceDecision` with `final_confidence`, `urgency`, `confidence_gain`, peer stats |
| Side effect | Publishes a `ZhiTag` to peers; receives peer `ZhiTag`s in background |
| Privacy | Only confidence and urgency are shared — no raw sensor data leaves the device |
| Overhead | Sub-kilobyte per second per node; runtime overhead < 2% vs baseline |
