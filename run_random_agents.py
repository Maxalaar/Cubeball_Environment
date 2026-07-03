from pathlib import Path
from cubeball import Cubeball


def sample_random_actions(environment: Cubeball) -> dict:
    return {
        agent_name: environment.action_spaces[agent_name].sample()
        for agent_name in environment.agents
    }


def main() -> None:
    environment_configuration = {
        # "render_mode": "human",
        "show_window": True,
        "action_repeat": 40,
        "speedup": 1.0,
    }

    environment = Cubeball(environment_configuration)

    try:
        while True:
            environment.reset()
            done = False

            while not done:
                actions = sample_random_actions(environment)
                _, _, dones, _, _ = environment.step(actions)
                done = all(dones.values())
    except KeyboardInterrupt:
        pass
    finally:
        environment.close()


if __name__ == "__main__":
    main()
