import dataclasses
import datetime
import hashlib
import importlib.metadata
import pathlib
import platform
import re
import subprocess
import time

import yaml
import matplotlib.pyplot as plt

from cubeball import Cubeball, GameModeRange, GameModeTeamRange
from cubeball.environment import GAME_EXECUTABLE_PATH
from cubeball.reward_functions.goal_reward import GoalReward

WARMUP_STEPS = 200
BENCHMARK_STEPS = 2000
PRINT_EVERY = 200

RESULTS_ROOT = pathlib.Path(__file__).parent / "benchmark_results"


def sample_random_actions(environment: Cubeball) -> dict:
    return {
        agent_name: environment.action_spaces[agent_name].sample()
        for agent_name in environment.agents
    }


def run_steps(environment: Cubeball, num_steps: int, on_step=None) -> int:
    episodes_completed = 0
    environment.reset()

    for _ in range(num_steps):
        actions = sample_random_actions(environment)
        _, _, dones, _, _ = environment.step(actions)

        if all(dones.values()):
            episodes_completed += 1
            environment.reset()

        if on_step is not None:
            on_step()

    return episodes_completed


def get_godot_version() -> str:
    try:
        output = subprocess.run(
            [GAME_EXECUTABLE_PATH, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return output.stdout.strip() or output.stderr.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def get_godot_binary_info() -> dict:
    # The engine version alone (get_godot_version) doesn't change between exports of the
    # same Godot version, so it can't tell apart two different builds of the game itself.
    # The .pck holds the exported project content, so hashing it identifies the actual build.
    pck_path = pathlib.Path(GAME_EXECUTABLE_PATH).with_suffix(".pck")
    try:
        pck_sha256 = hashlib.sha256(pck_path.read_bytes()).hexdigest()
        modified_at = datetime.datetime.fromtimestamp(pck_path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        pck_sha256 = "unknown"
        modified_at = "unknown"
    return {"pck_sha256": pck_sha256, "pck_modified_at": modified_at}


def get_gpu_info() -> str:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if output.returncode == 0 and output.stdout.strip():
            return output.stdout.strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def get_cpu_info() -> str:
    try:
        cpuinfo = pathlib.Path("/proc/cpuinfo").read_text()
        match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        model_name = match.group(1).strip() if match else platform.processor() or "unknown"
    except OSError:
        model_name = platform.processor() or "unknown"
    return f"{model_name} ({os_cpu_count()} logical cores)"


def os_cpu_count() -> int:
    import os
    return os.cpu_count() or 0


def get_ram_info() -> str:
    try:
        meminfo = pathlib.Path("/proc/meminfo").read_text()
        match = re.search(r"MemTotal:\s*(\d+)\s*kB", meminfo)
        if match:
            return f"{int(match.group(1)) / 1024 ** 2:.1f} GB"
    except OSError:
        pass
    return "unknown"


def get_package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def collect_system_info() -> dict:
    godot_binary_info = get_godot_binary_info()
    return {
        "execution_datetime": datetime.datetime.now().isoformat(timespec="seconds"),
        "godot_engine_version": get_godot_version(),
        "godot_binary_pck_sha256": godot_binary_info["pck_sha256"],
        "godot_binary_modified_at": godot_binary_info["pck_modified_at"],
        "os": platform.platform(),
        "cpu": get_cpu_info(),
        "ram": get_ram_info(),
        "gpu": get_gpu_info(),
        "python_version": platform.python_version(),
        "package_versions": {
            package_name: get_package_version(package_name)
            for package_name in ["numpy", "torch", "ray", "gymnasium"]
        },
    }


def serialize_environment_configuration(environment_configuration: dict) -> dict:
    serialized = dict(environment_configuration)
    serialized["reward_function"] = environment_configuration["reward_function"].__name__
    serialized["game_mode_range"] = dataclasses.asdict(environment_configuration["game_mode_range"])
    return serialized


def save_results_plot(output_path: pathlib.Path, time_series: list[dict], average_steps_per_second: float) -> None:
    steps_done = [entry["steps_done"] for entry in time_series]
    steps_per_second = [entry["steps_per_second"] for entry in time_series]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(steps_done, steps_per_second, marker="o", label="steps/s (interval)")
    axis.axhline(average_steps_per_second, color="red", linestyle="--", label="average")
    axis.set_xlabel("Env steps")
    axis.set_ylabel("Steps/s")
    axis.set_title("Cubeball environment throughput")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    environment_configuration = {
        "show_window": True,
        "action_repeat": 8,
        # Uncapped from real-time (speedup scales Engine.time_scale) so the benchmark
        # measures actual hardware throughput instead of being capped at 1x real-time.
        "speedup": 20.0,
        "debug_logs": False,
        "observation_mode": "token",

        "disable_cameras": False,
        "disable_environment": False,
        "display_fps": True,

        "reward_function": GoalReward,

        "game_mode_range": GameModeRange(
            level_size=((10, 4, 15), (20, 4, 30)),
            goal_size=((3, 4, 5), (3, 4, 5)),
            cuboid_field_margin=((0, 0, 0), (0, 0, 0)),
            ball_number=(1, 2),
            obstacle_number=(0, 0),
            max_duration_seconds=(10, 20),
            max_goal=(1, 1),
            team_list=[
                GameModeTeamRange(players_number=(1, 3)),
                GameModeTeamRange(players_number=(1, 3)),
            ],
        ),
    }

    run_directory = RESULTS_ROOT / datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_directory.mkdir(parents=True, exist_ok=True)

    system_info = collect_system_info()

    environment = Cubeball(environment_configuration)

    try:
        print(f"Warming up ({WARMUP_STEPS} steps)...")
        run_steps(environment, WARMUP_STEPS)

        print(f"Benchmarking ({BENCHMARK_STEPS} steps)...")
        steps_done = 0
        time_series = []
        start_time = time.perf_counter()
        last_print_time = start_time

        def on_step():
            nonlocal steps_done, last_print_time
            steps_done += 1
            if steps_done % PRINT_EVERY == 0:
                now = time.perf_counter()
                interval_sps = PRINT_EVERY / (now - last_print_time)
                last_print_time = now
                time_series.append({"steps_done": steps_done, "steps_per_second": interval_sps})
                print(f"  {steps_done}/{BENCHMARK_STEPS} steps - {interval_sps:.1f} steps/s")

        episodes_completed = run_steps(environment, BENCHMARK_STEPS, on_step=on_step)
        elapsed_seconds = time.perf_counter() - start_time

        steps_per_second = BENCHMARK_STEPS / elapsed_seconds
        simulated_ticks_per_second = steps_per_second * environment_configuration["action_repeat"]
        interval_values = [entry["steps_per_second"] for entry in time_series]

        results = {
            "benchmark_steps": BENCHMARK_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "elapsed_seconds": elapsed_seconds,
            "episodes_completed": episodes_completed,
            "steps_per_second_average": steps_per_second,
            "steps_per_second_min": min(interval_values),
            "steps_per_second_max": max(interval_values),
            "simulated_physics_ticks_per_second_average": simulated_ticks_per_second,
            "time_series": time_series,
        }

        print()
        print("Results:")
        print(f"  Elapsed time:         {elapsed_seconds:.2f} s")
        print(f"  Env steps:            {BENCHMARK_STEPS}")
        print(f"  Episodes completed:   {episodes_completed}")
        print(f"  Steps/s:              {steps_per_second:.1f}")
        print(f"  Simulated physics ticks/s (steps/s * action_repeat): {simulated_ticks_per_second:.1f}")

        with open(run_directory / "results.yaml", "w") as file:
            yaml.safe_dump(results, file, sort_keys=False)
        save_results_plot(run_directory / "results.png", time_series, steps_per_second)

        with open(run_directory / "environment_configuration.yaml", "w") as file:
            yaml.safe_dump(serialize_environment_configuration(environment_configuration), file, sort_keys=False)

        with open(run_directory / "system_info.yaml", "w") as file:
            yaml.safe_dump(system_info, file, sort_keys=False)

        print(f"\nSaved benchmark run to {run_directory}")
    except ConnectionError:
        print("Godot window closed, exiting.")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
