# Copilot Instructions — DriveLab

Full background lives in `CONTEXT.md` and `docs/PROJECT_STATE.md` (or wherever the detailed
state doc is kept) at the repo root. This file is the short, always-on behavioral guide.

## Project in one line
End-to-end behavioral-cloning: forward RGB camera image -> steer, throttle, brake.
CARLA's Traffic Manager drives first and generates the demonstration dataset; a small CNN
then learns to approximate those actions from pixels alone. Not a full production AV stack.

## Author context (affects code style, not just content)
9+ years C++ systems/real-time 3D engineer, pivoting to AV infrastructure roles.
Comfortable with CARLA/PyTorch/OpenCV now after building the initial pipeline.
- Prefer clean, explicit, inspectable code over clever abstractions.
- Prefer small, verifiable steps over large speculative scaffolding.
- Comment non-obvious CARLA/PyTorch API usage, especially anything version-specific.

## Environment (do not assume defaults — this is the real setup)
- CARLA server 0.9.16, run separately as CarlaUE4.exe (not started from this code).
- Python 3.12 venv at `.venv`.
- torch 2.11.0+cu128, CUDA-enabled; RTX 4070, 12GB VRAM. Training auto-uses `cuda`, falls
  back to CPU.
- Deps in requirements.txt: carla, numpy, pandas, opencv-python, pyyaml, torch, torchvision.
- `config.yaml` is the single source of truth for a standard run (host/port, map, sync tick
  rate, ego blueprint, spawn point, NPC count, camera resolution/FOV, output paths).
  Prefer editing config.yaml over hardcoding new run parameters in scripts.
- `connect.py` supports CLI overrides (`--frames`, `--scenario-id`); follow that pattern for
  new run-time parameters instead of inventing a separate mechanism.

## Current milestone — stay scoped to this
Data collection, dataset prep, and a smoke-test training loop are DONE (see "Implemented"
below). The active next step is:

**Add deliberately controlled obstacle scenarios (stalled vehicle, cones, slow lead vehicle),
then collect them as distinct, correctly labeled train/validation scenario sets.**

Do not propose closed-loop learned-policy driving, multi-epoch tuning, or evaluation metrics
until obstacle-scenario data exists — those depend on it. Suggest them as "later" notes only.

## Implemented (don't redo, extend carefully)
- CARLA connection, map loading, synchronous-mode ticking (world.tick() per timestep).
- Ego vehicle + RGB camera + autopilot NPC traffic spawn/teardown (connect.py).
- CSV manifest capture: image_path, timestamp, steer, throttle, brake, speed_kmh,
  scenario_id, split.
- viz/inspect_manifest.py (pandas stats, steer histogram) and viz/stitch_video.py
  (overlay + MP4 preview).
- dataset.py: DrivingDataset — PNG -> RGB -> resize 800x600 to 256x192 -> (3,192,256)
  float32 tensor normalized to ~[-1,1]; target tensor [steer, throttle, brake].
  scenario_ids param used to split by session, not by random frame.
- model.py: DrivingPolicy CNN (Conv 3->16->32->64, GAP, FC 64->32->3), ~26.5K params,
  regression not classification.
- train.py: MSE loss, Adam optimizer, GPU training, scenario-held-out validation,
  checkpoint save/load (smoke_test.pt), verified to reload into a fresh DrivingPolicy.

## Not implemented yet — real gaps, be aware when suggesting code
- Scripted obstacle scenarios (stalled vehicle, cones, slow lead vehicle).
- Weather / time-of-day variation.
- Real train/val labels in the `split` column (currently all "train"; split is faked via
  scenario_id in train.py).
- Steering-imbalance handling (e.g., weighted sampling) — histogram shows heavy near-zero
  steering skew from straight-road driving.
- Multi-epoch experiments with recorded loss curves.
- Predicted-vs-ground-truth inspection tooling.
- Closed-loop evaluation: running the trained policy in CARLA instead of just offline MSE.
- Collision count, route completion, intervention metrics.

## Known non-issues
Pylance warnings on `set_autopilot()` / actor return types in connect.py are due to CARLA's
incomplete type stubs (labels `Vehicle`/`Sensor` as generic `Actor`), not real runtime bugs.
Don't "fix" these by restructuring working code — narrow casts/type comments are enough.

## Non-goals right now
No LiDAR/radar fusion, no ROS2 integration, no RL-from-scratch, no multi-map generalization
study, no production-grade dashboard.

## Efficiency notes for suggestions
- Keep generated code minimal and directly runnable; avoid speculative config options or
  generalized frameworks not yet needed.
- When unsure about intent, ask a clarifying question instead of generating a large,
  possibly-wrong implementation.
- Follow existing patterns (config.yaml driven, CLI overrides, scenario_id labeling) rather
  than introducing a new configuration style for new features.
