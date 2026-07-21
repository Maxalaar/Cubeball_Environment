from dataclasses import dataclass

import numpy as np

Vector3 = tuple[float, float, float]


@dataclass
class GameModeTeam:
    players_number: int


# Mirrors scripts/game_mode.gd's GameMode resource field-for-field: this is the concrete
# configuration of a single match, converted to the wire format Godot's
# GameModeManager.apply_overrides expects (see to_config).
@dataclass
class GameMode:
    level_size: Vector3
    goal_size: Vector3
    ball_number: int
    obstacle_number_min: int
    obstacle_number_max: int
    max_duration_seconds: float
    max_goal: int
    team_list: list[GameModeTeam]

    def to_config(self) -> dict:
        return {
            "level_size": list(self.level_size),
            "goal_size": list(self.goal_size),
            "ball_number": self.ball_number,
            "obstacle_number_min": self.obstacle_number_min,
            "obstacle_number_max": self.obstacle_number_max,
            "max_duration_seconds": self.max_duration_seconds,
            "max_goal": self.max_goal,
            "players_per_team": [team.players_number for team in self.team_list],
        }


@dataclass
class GameModeTeamRange:
    players_number: tuple[int, int]

    def sample(self, rng: np.random.Generator) -> GameModeTeam:
        return GameModeTeam(players_number=int(rng.integers(self.players_number[0], self.players_number[1] + 1)))


# Same shape as GameMode, but every field is a (min, max) range instead of a concrete
# value — a fixed value is just a degenerate range like (5, 5). Cubeball samples a fresh
# GameMode from this every episode (see GameMode.sample below / Cubeball.reset).
@dataclass
class GameModeRange:
    level_size: tuple[Vector3, Vector3]
    goal_size: tuple[Vector3, Vector3]
    ball_number: tuple[int, int]
    obstacle_number_min: tuple[int, int]
    obstacle_number_max: tuple[int, int]
    max_duration_seconds: tuple[float, float]
    max_goal: tuple[int, int]
    team_list: list[GameModeTeamRange]

    @property
    def max_players_per_team(self) -> list[int]:
        return [team.players_number[1] for team in self.team_list]

    def sample(self, rng: np.random.Generator) -> GameMode:
        return GameMode(
            level_size=tuple(float(rng.uniform(low, high)) for low, high in zip(*self.level_size)),
            goal_size=tuple(float(rng.uniform(low, high)) for low, high in zip(*self.goal_size)),
            ball_number=int(rng.integers(self.ball_number[0], self.ball_number[1] + 1)),
            obstacle_number_min=int(rng.integers(self.obstacle_number_min[0], self.obstacle_number_min[1] + 1)),
            obstacle_number_max=int(rng.integers(self.obstacle_number_max[0], self.obstacle_number_max[1] + 1)),
            max_duration_seconds=float(rng.uniform(*self.max_duration_seconds)),
            max_goal=int(rng.integers(self.max_goal[0], self.max_goal[1] + 1)),
            team_list=[team.sample(rng) for team in self.team_list],
        )
