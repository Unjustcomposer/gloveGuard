# GloveGuard: Multimodal Telemetry QA & PyTorch Ingestion Engine

**Automated hardware-fault detection, cross-modal clock alignment, and zero-copy tensor sharding for high-scale robotics demonstration factories.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Polars](https://img.shields.io/badge/Vectorized_QA-Polars-orange.svg)](https://pola.rs/)
[![PyTorch](https://img.shields.io/badge/Ingestion-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Rerun.io](https://img.shields.io/badge/Visualizer-Rerun.io-00b4d8.svg)](https://rerun.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Overview

When scaling human demonstration collection across sensor-glove robotics rigs, **silent hardware data corruption** is the primary bottleneck to foundation model generalization. A single loose flex-sensor wire, USB frame drop, or clock drift event during an 8-hour shift can poison thousands of action-conditioning trajectories before reaching the training cluster.

`GloveGuard` sits between raw factory collection rigs (30 Hz ego-camera video + 200 Hz 6-axis IMU + 200 Hz 5-finger tactile force glove) and PyTorch foundation model training—catching dead channels, kinematic spikes, and camera-to-IMU desynchronization in `<15ms` per episode using vectorized Polars execution and zero-copy contiguous 64-step tensor sharding.

---

## System Architecture

```mermaid
flowchart LR
    subgraph Factory_Rig ["Data Collection Rig (Raw Streams)"]
        V["Ego-Camera Stream<br/>(30 Hz RGB)"]
        I["6-Axis Wrist IMU<br/>(200 Hz Accel/Gyro)"]
        F["5-Finger Tactile Glove<br/>(200 Hz Force/ADC)"]
    end

    subgraph GloveGuard ["GloveGuard Ingestion & QA Gate (Polars + OpenCV)"]
        A["Backward Asof Clock Aligner<br/>(Sub-5ms Jitter Tolerance)"]
        G1["Gate 1: ADC Dead-Zone Detector<br/>(100ms Rolling σ < 1e-4 N)"]
        G2["Gate 2: Kinematic Plausibility<br/>(‖a‖₂ > 45 m/s² Hardware Bump)"]
        G3["Gate 3: Cross-Modal Optical Flow<br/>(Frozen Video vs. Active IMU)"]
        D{"Corruption Ratio<br/>< 30%?"}
    end

    subgraph Downstream ["Downstream Consumers"]
        S["Window-Level Excision<br/>(64-Step Contiguous Chunks)"]
        P[("PyTorch IterableDataset<br/>Training Shards")]
        Q[("Quarantine Bucket<br/>+ Rig Alert JSON")]
        R["Rerun.io 4-Panel<br/>Multimodal Inspector"]
    end

    V & I & F --> A
    A --> G1 & G2 & G3
    G1 & G2 & G3 --> D
    D -- "PASS (Excise Bad Windows)" --> S --> P
    D -- "FAIL (Reject Episode)" --> Q
    G1 & G2 & G3 -. "Live Telemetry Stream" .-> R
```

---

## Injected Hardware Faults & Quality Gates

| Fault Type | Target Channel | Time Window | Physical Mechanism | Vectorized Detection Gate |
|---|---|---|---|---|
| **Sensor Flatline** | `force_index` | 2.5s – 3.5s | Saturated ADC / loose flex-sensor wire | `rolling_std(window=20) < 1e-4 N` |
| **Kinematic Spike** | `acc_x`, `acc_y` | 5.8s – 5.95s | Sensor impact / wrist hitting table | $L_2\text{-norm}(\mathbf{a}) > 45.0\,\text{m/s}^2$ |
| **Cross-Modal Desync** | Ego Video | 7.5s – 8.3s | USB packet drop / camera freeze | $v_{\text{wrist}} > 0.15\,\text{m/s} \land \Delta_{\text{flow}} < 0.01$ |

---

## Clean Repository Structure

```text
.
├── gloveguard/
│   ├── __init__.py          # Package entrypoint & API exports
│   ├── simulator.py         # 30Hz video + 200Hz IMU/Glove generator with 3 injected faults
│   ├── validator.py         # Vectorized Polars QA engine + optical-flow sync gate
│   ├── sharder.py           # PyTorch IterableDataset + clean window Parquet/Tensor shard exporter
│   └── visualizer.py        # Rerun.io 4-panel multimodal dashboard with live QA HUD overlays
├── tests/
│   └── test_validator.py    # Pytest regression suite (100% anomaly recall + zero corrupt shards)
├── artifacts/
│   ├── shards/              # Clean [64, 11] float32 tensor shards (.npy, .parquet) + manifest
│   ├── qa_report.json       # Telemetry QA summary report
│   └── gloveguard_dashboard.rrd # Headless Rerun dashboard recording
├── run_pipeline.py          # End-to-end CLI pipeline runner
├── requirements.txt         # Pinned production dependencies
├── pyproject.toml           # Pytest & package discovery configuration
├── .gitignore               # Production git exclusion rules
└── README.md                # Project documentation
```

---

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Full QA Pipeline
```bash
# Interactive mode (opens Rerun 4-panel dashboard)
python run_pipeline.py

# Headless mode (CI/CD / server export to ./artifacts/)
python run_pipeline.py --headless

# Custom duration and seed
python run_pipeline.py --duration 12.0 --seed 123 --headless
```

### 3. Run Test Suite
```bash
pytest tests/ -v
```

---

## CLI Options

```text
usage: run_pipeline.py [-h] [--duration DURATION] [--seed SEED] [--headless] [--export-dir EXPORT_DIR]

GloveGuard end-to-end pipeline

options:
  -h, --help            show this help message and exit
  --duration DURATION   Episode duration in seconds (default: 10.0)
  --seed SEED           RNG seed for deterministic generation (default: 42)
  --headless            Skip Rerun viewer; save .rrd to export dir instead
  --export-dir EXPORT_DIR
                        Root directory for shard and dashboard exports (default: ./artifacts)
```

---

## Technical Design Trade-offs

1. **Vectorized Polars vs. Python Iteration:**  
   Evaluating 200Hz multi-channel telemetry with rolling standard deviation and Euclidean norms in Python loops creates unacceptable latency ($>300\,\text{ms}$). Using Polars vectorized columnar expressions executes all gates in under $15\,\text{ms}$, enabling real-time edge QA on factory capture rigs.

2. **Optical-Flow Proxy vs. Dense Flow (RAFT/Farnebäck):**  
   We calculate the mean absolute grayscale frame difference between consecutive frames ($\approx 0.5\,\text{ms}$ per frame) rather than computing expensive dense optical flow fields. This proxy reliably catches camera freezes and dropped frames while maintaining a sub-10ms validation budget.

3. **Window-Level Hard Excision vs. Interpolation:**  
   Corrupted 64-step windows are strictly discarded rather than interpolated. Imputing or interpolating through physical sensor faults (such as violent kinematic spikes or frozen cameras) teaches imitation models physically invalid dynamics.

4. **Backward `asof` Timestamp Synchronization:**  
   Sensor samples at 200Hz and camera frames at 30Hz arrive with clock skew and jitter. `pl.DataFrame.join_asof(strategy="backward")` maps each 200Hz telemetry row to the most recently acquired camera frame without lookahead bias.

---

## License

MIT License. Designed for robust robotics data infrastructure.
