from utilities import default_environment_configuration, run_environment, run_steps


def main() -> None:
    environment_configuration = default_environment_configuration(
        show_window=True,
        speedup=1.0,
        disable_post_goal_duration=False,
        disable_ui=False,
        disable_goal_nets=False,
        disable_cameras=False,
        disable_environment=False,
        debug_logs=True,
    )

    with run_environment(environment_configuration) as environment:
        try:
            run_steps(
                environment,
                on_episode_start=lambda: print("Active agents this episode:", environment.agents),
            )
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
