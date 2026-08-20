from utilities import default_environment_configuration, run_environment, run_steps


def main() -> None:
    environment_configuration = default_environment_configuration(
        speedup=1.0,
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
