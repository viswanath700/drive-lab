# DriveLab

Simulation-based driving policy fine-tuning in CARLA: an imitation-learning policy
fine-tuned on scripted obstacle-avoidance scenarios, with a full data pipeline and
evaluation/visualization layer. This is **not** a full self-driving stack — see
[Limitations](#limitations).

## Setup

1. Install and run a CARLA server (e.g. `CarlaUE4.exe` on Windows), default port `2000`.
2. Create a virtual environment and install dependencies:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   The `carla` pip package version **must match your CARLA server version exactly**.
   If it's not on PyPI for your version, install the `.whl` shipped in the server's
   `PythonAPI/carla/dist/` folder instead.
3. Adjust [configs/config.yaml](configs/config.yaml) if needed (town, vehicle, camera,
   capture settings).

## Usage

With the CARLA server running:

```powershell
python -m sim.connect --verbose
```

- Connects to the server, prints the map name, and (with `--verbose`) lists vehicle
  blueprints and spawn points.
- Spawns an ego vehicle with autopilot enabled.
- Attaches a forward-facing RGB camera.
- Runs the simulation in **synchronous mode** (fixed tick length) so each saved frame
  and its logged control/speed values come from the same simulation tick.
- Saves frames to `data/raw/frame_XXXXXX.png` and appends one row per frame to
  `data/manifest.csv` (`image_path, timestamp, steer, throttle, brake, speed_kmh,
  scenario_id, split`).
- Cleans up (destroys actors, restores async mode) on exit, including on Ctrl+C.

Useful flags:
```powershell
python -m sim.connect --config configs/config.yaml --frames 500
```

Load and inspect the manifest with pandas:
```python
import pandas as pd
df = pd.read_csv("data/manifest.csv")
```

## Repo structure

```
sim/         - CARLA connection + spawn/sensor scripts, scenario scripts
data/        - CSV manifests, not large binaries (raw frames are gitignored)
training/    - dataset class, model, train loop, checkpoints
evaluation/  - route runner, metrics (collisions, completion, interventions)
viz/         - overlay plots, video stitching
configs/     - YAML config shared by all scripts
```

## Current status

- [x] Connect to CARLA, print map name, list blueprints/spawn points.
- [x] Spawn ego vehicle with autopilot.
- [x] Attach RGB camera, save frames to disk via synchronous-mode capture loop.
- [x] Log per-frame controls/speed/timestamp to CSV alongside images.
- [ ] Preview saved frames / stitch into short video with OpenCV.
- [x] Clean actor teardown on exit.

## Limitations

- Plain behavioral cloning (no DAgger), so compounding error / distributional shift
  under real deployment conditions is expected and not addressed here.
- No LiDAR/radar fusion, no ROS2 integration, no RL-from-scratch, no multi-map
  generalization study, no production-grade dashboard.
- Results are scoped to the scripted scenarios and town(s) used for data collection.
