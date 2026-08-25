"""Core data-collection loop: connect to CARLA, spawn an ego vehicle, capture RGB
frames with per-frame control/speed logged to a CSV manifest, then tear everything
down cleanly.

Usage:
    python -m sim.connect
    python -m sim.connect --config configs/config.yaml --frames 500 --verbose

Runs the simulation in *synchronous* mode so that each saved image and its
logged control/speed values come from the exact same simulation tick. Relying
on the default asynchronous mode with a threaded sensor callback can silently
misalign frames and labels, which quietly corrupts an imitation-learning
dataset.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import queue
import random
import sys
from pathlib import Path

try:
    import carla
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit(
        "Could not import the 'carla' package. Install the wheel that matches "
        "your CARLA server version (see requirements.txt)."
    ) from exc

from sim.config import REPO_ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drivelab.sim.connect")

MANIFEST_FIELDS = [
    "image_path",
    "timestamp",
    "steer",
    "throttle",
    "brake",
    "speed_kmh",
    "scenario_id",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML.")
    parser.add_argument("--frames", type=int, default=None, help="Override capture.max_frames.")
    parser.add_argument(
        "--scenario-id", type=str, default=None, help="Override capture.scenario_id."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="List available blueprints and spawn points."
    )
    return parser.parse_args()


def connect_and_get_world(carla_cfg: dict) -> tuple["carla.Client", "carla.World"]:
    client = carla.Client(carla_cfg["host"], carla_cfg["port"])
    client.set_timeout(carla_cfg["timeout_sec"])

    world = client.get_world()
    town = carla_cfg.get("town")
    if town and world.get_map().name.split("/")[-1] != town:
        logger.info("Loading map %s (current map is %s)...", town, world.get_map().name)
        world = client.load_world(town)

    logger.info("Connected. Current map: %s", world.get_map().name)
    return client, world


def enable_synchronous_mode(client: "carla.Client", world: "carla.World", fixed_delta_seconds: float):
    original_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    return original_settings, traffic_manager


def log_blueprints_and_spawn_points(world: "carla.World") -> None:
    blueprint_library = world.get_blueprint_library()
    vehicle_bps = [bp.id for bp in blueprint_library.filter("vehicle.*")]
    logger.info("Available vehicle blueprints (%d): %s", len(vehicle_bps), vehicle_bps)

    spawn_points = world.get_map().get_spawn_points()
    logger.info("Available spawn points: %d", len(spawn_points))


def spawn_ego_vehicle(world: "carla.World", vehicle_cfg: dict) -> "carla.Vehicle":
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find(vehicle_cfg["blueprint"])

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available on this map.")

    index = vehicle_cfg.get("spawn_point_index")
    spawn_point = spawn_points[index] if index is not None else random.choice(spawn_points)

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is None:
        raise RuntimeError(f"Failed to spawn vehicle at spawn point index {index}.")

    logger.info("Spawned ego vehicle %s (id=%d) at %s", vehicle_cfg["blueprint"], vehicle.id, spawn_point.location)

    if vehicle_cfg.get("autopilot", False):
        vehicle.set_autopilot(True)
        logger.info("Autopilot enabled.")

    return vehicle


def spawn_background_traffic(world: "carla.World", count: int) -> list:
    """Spawn NPC vehicles under autopilot so the ego encounters other traffic
    (lane-following, stopping distance, merges) instead of an empty road."""
    if count <= 0:
        return []

    blueprint_library = world.get_blueprint_library()
    vehicle_bps = blueprint_library.filter("vehicle.*")
    spawn_points = list(world.get_map().get_spawn_points())
    random.shuffle(spawn_points)

    traffic_actors = []
    for spawn_point in spawn_points:
        if len(traffic_actors) >= count:
            break
        npc = world.try_spawn_actor(random.choice(vehicle_bps), spawn_point)
        if npc is not None:
            npc.set_autopilot(True)
            traffic_actors.append(npc)

    logger.info("Spawned %d background traffic vehicles", len(traffic_actors))
    return traffic_actors


def attach_rgb_camera(
    world: "carla.World", vehicle: "carla.Vehicle", camera_cfg: dict
) -> "carla.Sensor":
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find(camera_cfg["blueprint"])
    camera_bp.set_attribute("image_size_x", str(camera_cfg["width"]))
    camera_bp.set_attribute("image_size_y", str(camera_cfg["height"]))
    camera_bp.set_attribute("fov", str(camera_cfg["fov"]))

    t = camera_cfg["transform"]
    transform = carla.Transform(
        carla.Location(x=t["x"], y=t["y"], z=t["z"]),
        carla.Rotation(pitch=t["pitch"], yaw=t["yaw"], roll=t["roll"]),
    )

    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    logger.info("Attached RGB camera (id=%d) at %s", camera.id, transform)
    return camera


def vehicle_speed_kmh(vehicle: "carla.Vehicle") -> float:
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)


def run_capture_loop(
    world: "carla.World",
    vehicle: "carla.Vehicle",
    camera: "carla.Sensor",
    capture_cfg: dict,
) -> None:
    output_dir = REPO_ROOT / capture_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = REPO_ROOT / capture_cfg["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()

    image_queue: "queue.Queue" = queue.Queue()
    camera.listen(image_queue.put)

    max_frames = capture_cfg["max_frames"]
    scenario_id = capture_cfg["scenario_id"]
    split = capture_cfg["split"]

    logger.info("Starting capture loop: %d frames -> %s", max_frames, output_dir)

    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()

        for i in range(max_frames):
            frame_id = world.tick()
            image = image_queue.get()

            control = vehicle.get_control()
            speed = vehicle_speed_kmh(vehicle)
            timestamp = world.get_snapshot().timestamp.elapsed_seconds

            image_filename = f"frame_{frame_id:06d}.png"
            image_path = output_dir / image_filename
            image.save_to_disk(str(image_path))

            writer.writerow(
                {
                    "image_path": str(image_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "timestamp": f"{timestamp:.6f}",
                    "steer": f"{control.steer:.6f}",
                    "throttle": f"{control.throttle:.6f}",
                    "brake": f"{control.brake:.6f}",
                    "speed_kmh": f"{speed:.4f}",
                    "scenario_id": scenario_id,
                    "split": split,
                }
            )

            if (i + 1) % 100 == 0 or (i + 1) == max_frames:
                logger.info("Captured %d/%d frames", i + 1, max_frames)

    camera.stop()
    logger.info("Capture loop finished. Manifest: %s", manifest_path)


def teardown(
    client: "carla.Client",
    world: "carla.World",
    original_settings: "carla.WorldSettings",
    traffic_manager: "carla.TrafficManager",
    actors: list,
) -> None:
    logger.info("Tearing down actors and restoring world settings...")
    for actor in actors:
        if actor is not None and actor.is_alive:
            actor.destroy()

    traffic_manager.set_synchronous_mode(False)
    world.apply_settings(original_settings)
    logger.info("Teardown complete.")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.frames is not None:
        config["capture"]["max_frames"] = args.frames
    if args.scenario_id is not None:
        config["capture"]["scenario_id"] = args.scenario_id

    client, world = connect_and_get_world(config["carla"])

    if args.verbose:
        log_blueprints_and_spawn_points(world)

    original_settings, traffic_manager = enable_synchronous_mode(
        client, world, config["carla"]["fixed_delta_seconds"]
    )

    vehicle = None
    camera = None
    traffic_actors: list = []
    try:
        vehicle = spawn_ego_vehicle(world, config["vehicle"])
        camera = attach_rgb_camera(world, vehicle, config["camera"])
        traffic_actors = spawn_background_traffic(
            world, config.get("traffic", {}).get("num_vehicles", 0)
        )
        run_capture_loop(world, vehicle, camera, config["capture"])
    finally:
        teardown(client, world, original_settings, traffic_manager, [camera, vehicle, *traffic_actors])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(1)
