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
        "--scenario",
        type=str,
        default=None,
        help=(
            "Name of a configs/config.yaml scenarios: entry (e.g. obstacle_cone_session1). "
            "Overrides the ego spawn transform and spawns the scenario's obstacle actor(s). "
            "Also defaults capture.scenario_id to this name unless --scenario-id is given."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val"],
        help="Override capture.split.",
    )
    parser.add_argument(
        "--weather",
        type=str,
        default=None,
        help="Override weather.preset with a carla.WeatherParameters name (e.g. ClearNoon, WetNoon, HardRainSunset).",
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


def apply_weather(world: "carla.World", weather_cfg: dict) -> None:
    """Set world weather from a named carla.WeatherParameters preset (e.g. ClearNoon).

    Weather is a world-level property, not tied to the ego vehicle, so this only
    needs to run once per session before capture starts.
    """
    preset_name = (weather_cfg or {}).get("preset")
    if not preset_name:
        return

    preset = getattr(carla.WeatherParameters, preset_name, None)
    if preset is None:
        raise ValueError(
            f"Unknown weather preset '{preset_name}'. Must match a carla.WeatherParameters "
            "attribute, e.g. ClearNoon, CloudyNoon, WetNoon, HardRainSunset."
        )
    world.set_weather(preset)
    logger.info("Weather set to preset '%s'", preset_name)


def log_blueprints_and_spawn_points(world: "carla.World") -> None:
    blueprint_library = world.get_blueprint_library()
    vehicle_bps = [bp.id for bp in blueprint_library.filter("vehicle.*")]
    logger.info("Available vehicle blueprints (%d): %s", len(vehicle_bps), vehicle_bps)

    spawn_points = world.get_map().get_spawn_points()
    logger.info("Available spawn points: %d", len(spawn_points))


def spawn_ego_vehicle(world: "carla.World", vehicle_cfg: dict) -> "carla.Vehicle":
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find(vehicle_cfg["blueprint"])

    spawn_transform_cfg = vehicle_cfg.get("spawn_transform")
    if spawn_transform_cfg is not None:
        spawn_point = build_transform(spawn_transform_cfg, world)
    else:
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available on this map.")
        index = vehicle_cfg.get("spawn_point_index")
        spawn_point = spawn_points[index] if index is not None else random.choice(spawn_points)

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is None:
        raise RuntimeError(f"Failed to spawn vehicle at {spawn_point.location}.")

    logger.info("Spawned ego vehicle %s (id=%d) at %s", vehicle_cfg["blueprint"], vehicle.id, spawn_point.location)

    if vehicle_cfg.get("autopilot", False):
        vehicle.set_autopilot(True)
        logger.info("Autopilot enabled.")

    return vehicle


def build_transform(transform_cfg: dict, world: "carla.World") -> "carla.Transform":
    """Build a carla.Transform from explicit x/y/z/yaw config, snapping z to the
    road surface at that (x, y) so obstacle/ego actors don't spawn underground
    or floating — only x/y/yaw come from the config, height comes from the map.
    """
    location = carla.Location(x=transform_cfg["x"], y=transform_cfg["y"])
    waypoint = world.get_map().get_waypoint(location, project_to_road=True)
    location.z = waypoint.transform.location.z + transform_cfg.get("z", 0.3)
    rotation = carla.Rotation(yaw=transform_cfg.get("yaw", 0.0))
    return carla.Transform(location, rotation)


def spawn_obstacles(
    world: "carla.World", obstacles_cfg: list[dict], traffic_manager: "carla.TrafficManager"
) -> list:
    """Spawn the obstacle actor(s) for a scripted scenario (stalled vehicle,
    cone/barrier, or slow lead vehicle) per configs/config.yaml's scenarios: block.
    """
    blueprint_library = world.get_blueprint_library()
    actors = []
    for obstacle_cfg in obstacles_cfg:
        bp_id = obstacle_cfg["blueprint"]
        blueprint = blueprint_library.find(bp_id)
        transform = build_transform(obstacle_cfg, world)

        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            raise RuntimeError(f"Failed to spawn obstacle '{bp_id}' at {transform.location}.")
        logger.info("Spawned obstacle %s (id=%d) at %s", bp_id, actor.id, transform.location)

        if bp_id.startswith("vehicle."):
            if obstacle_cfg.get("autopilot", False):
                actor.set_autopilot(True, traffic_manager.get_port())
                target_speed = obstacle_cfg.get("target_speed_kmh")
                if target_speed is not None:
                    traffic_manager.set_desired_speed(actor, target_speed)
            else:
                # No autopilot and no control ever applied would leave the vehicle
                # coasting/settling under physics; hand_brake keeps it truly stationary
                # for the whole capture (used for the stalled-vehicle obstacle type).
                actor.apply_control(carla.VehicleControl(hand_brake=True))

        actors.append(actor)

    return actors


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

    # skip first few frames to avoid unrealistic motion during the vehicle spawn
    warmup_ticks = capture_cfg["warmup_ticks"]
    logger.info("Warming up for %d ticks (vehicle settling)...", warmup_ticks)
    for _ in range(warmup_ticks):
        world.tick()
        image_queue.get(timeout=2.0)  # drain and discard — don't record

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
            image = image_queue.get(timeout=2.0) # Added timeout, so a stuck run fails loudly
            if image.frame != frame_id:
                logger.warning("Frame mismatch: expected %d, got %d — skipping", frame_id, image.frame)
                continue

            # Read control AFTER tick(), not before: this returns the control that was
            # actually applied during the physics step just completed, so it stays
            # aligned with the image/frame captured for that same tick.
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
            f.flush() # If this becomes a bottleneck, do it every 10-20 frames instead

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

    obstacles_cfg: list = []
    if args.scenario is not None:
        scenarios = config.get("scenarios", {})
        scenario_cfg = scenarios.get(args.scenario)
        if scenario_cfg is None:
            raise SystemExit(
                f"Unknown scenario '{args.scenario}'. Available: {sorted(scenarios)}"
            )
        config["vehicle"]["spawn_transform"] = scenario_cfg["ego_spawn"]
        obstacles_cfg = scenario_cfg.get("obstacles", [])
        if args.scenario_id is None:
            config["capture"]["scenario_id"] = args.scenario

    if args.frames is not None:
        config["capture"]["max_frames"] = args.frames
    if args.scenario_id is not None:
        config["capture"]["scenario_id"] = args.scenario_id
    if args.split is not None:
        config["capture"]["split"] = args.split
    if args.weather is not None:
        config.setdefault("weather", {})["preset"] = args.weather

    client, world = connect_and_get_world(config["carla"])
    apply_weather(world, config.get("weather", {}))

    if args.verbose:
        log_blueprints_and_spawn_points(world)

    original_settings, traffic_manager = enable_synchronous_mode(
        client, world, config["carla"]["fixed_delta_seconds"]
    )

    vehicle = None
    camera = None
    traffic_actors: list = []
    obstacle_actors: list = []
    try:
        vehicle = spawn_ego_vehicle(world, config["vehicle"])
        camera = attach_rgb_camera(world, vehicle, config["camera"])
        traffic_actors = spawn_background_traffic(
            world, config.get("traffic", {}).get("num_vehicles", 0)
        )
        obstacle_actors = spawn_obstacles(world, obstacles_cfg, traffic_manager)
        run_capture_loop(world, vehicle, camera, config["capture"])
    finally:
        teardown(
            client,
            world,
            original_settings,
            traffic_manager,
            [camera, vehicle, *traffic_actors, *obstacle_actors],
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(1)
