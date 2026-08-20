import contextlib

from cubeball import Cubeball, GameModeRange, GameModeTeamRange
from cubeball.reward_functions.goal_reward import GoalReward

DEFAULT_GAME_MODE_RANGE = GameModeRange(
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
)


def default_environment_configuration(**overrides) -> dict:
    configuration = {
        "show_window": True,
        "action_repeat": 8,
        "observation_mode": "token",
        "disable_cameras": False,
        "display_fps": True,
        "reward_function": GoalReward,
        "game_mode_range": DEFAULT_GAME_MODE_RANGE,
    }
    configuration.update(overrides)
    return configuration


def sample_random_actions(environment: Cubeball) -> dict:
    return {
        agent_name: environment.action_spaces[agent_name].sample()
        for agent_name in environment.agents
    }


def run_steps(environment: Cubeball, num_steps: int = None, on_step=None, on_episode_start=None) -> int:
    # num_steps=None runs until interrupted (e.g. KeyboardInterrupt), resetting into a new
    # episode each time the previous one ends -- used both for a fixed-length benchmark and
    # for an open-ended random-agent demo.
    episodes_completed = 0
    steps_done = 0

    environment.reset()
    if on_episode_start is not None:
        on_episode_start()

    while num_steps is None or steps_done < num_steps:
        actions = sample_random_actions(environment)
        _, _, dones, _, _ = environment.step(actions)
        steps_done += 1

        if all(dones.values()):
            episodes_completed += 1
            environment.reset()
            if on_episode_start is not None:
                on_episode_start()

        if on_step is not None:
            on_step()

    return episodes_completed


@contextlib.contextmanager
def run_environment(environment_configuration: dict):
    environment = Cubeball(environment_configuration)
    try:
        yield environment
    except ConnectionError:
        print("Godot window closed, exiting.")
    finally:
        environment.close()
