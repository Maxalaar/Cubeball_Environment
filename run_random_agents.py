from pathlib import Path
from cubeball import Cubeball
from game_mode import GameModeRange, GameModeTeamRange


def sample_random_actions(environment: Cubeball) -> dict:
    return {
        agent_name: environment.action_spaces[agent_name].sample()
        for agent_name in environment.agents
    }


def main() -> None:
    environment_configuration = {
        # "render_mode": "human",
        "show_window": True,
        "action_repeat": 8,
        "speedup": 1.0,
        "debug_logs": True,
        # Domain randomization: a fresh GameMode is sampled from this every episode
        # (see Cubeball.reset / GameModeRange.sample). A fixed value is just a
        # degenerate range, e.g. max_goal=(1, 1).

        "game_mode_range": GameModeRange(
            level_size=((10, 4, 15), (20, 4, 30)),
            goal_size=((3, 4, 5), (3, 4, 5)),
            ball_number=(1, 2),
            obstacle_number=(0, 0),
            max_duration_seconds=(10, 20),
            max_goal=(1, 1),
            team_list=[
                GameModeTeamRange(players_number=(1, 3)),
                GameModeTeamRange(players_number=(1, 3)),
            ],
        ),

        # "game_mode_range": GameModeRange(
        #     level_size=((10, 4, 15), (20, 4, 30)),
        #     goal_size=((3, 4, 5), (3, 4, 5)),
        #     ball_number=(1, 1),
        #     obstacle_number=(0, 0),
        #     max_duration_seconds=(10, 20),
        #     max_goal=(1, 1),
        #     team_list=[
        #         GameModeTeamRange(players_number=(1, 1)),
        #         GameModeTeamRange(players_number=(1, 1)),
        #     ],
        # ),
    }

    environment = Cubeball(environment_configuration)

    try:
        while True:
            environment.reset()
            print("Active agents this episode:", environment.agents)
            done = False

            while not done:
                actions = sample_random_actions(environment)
                _, _, dones, _, _ = environment.step(actions)
                done = all(dones.values())
    except KeyboardInterrupt:
        pass
    except ConnectionError:
        print("Godot window closed, exiting.")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
