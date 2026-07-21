from dataclasses import dataclass

import numpy as np

Vector3 = tuple[float, float, float]
Color = tuple[float, float, float, float]

# Mirrors resources/teams/*.tres in the Godot project — used to auto-assign a name/color
# to a team by index when GameModeTeamRange doesn't specify one, so existing game modes
# keep their usual look without every call site having to repeat these values.
DEFAULT_TEAM_PALETTE: list[tuple[str, Color]] = [
    ("Red Team", (0.5267932, 1.2153015e-06, 3.85046e-07, 1.0)),
    ("Blue Team", (3.5616756e-06, 0.0, 0.85208327, 1.0)),
    ("Green Team", (9.44566e-07, 0.34524155, 0.0869625, 1.0)),
    ("Yellow Team", (0.7407403, 0.6583144, 0.12537423, 1.0)),
]


@dataclass
class GameModeTeam:
    players_number: int
    name: str
    color: Color


# Mirrors scripts/game_mode.gd's GameMode resource field-for-field: this is the concrete
# configuration of a single match, converted to the wire format Godot's
# GameModeManager.create_game_mode expects (see to_config).
@dataclass
class GameMode:
    level_size: Vector3
    goal_size: Vector3
    ball_number: int
    obstacle_number: int
    max_duration_seconds: float
    max_goal: int
    team_list: list[GameModeTeam]

    def to_config(self) -> dict:
        return {
            "level_size": list(self.level_size),
            "goal_size": list(self.goal_size),
            "ball_number": self.ball_number,
            "obstacle_number": self.obstacle_number,
            "max_duration_seconds": self.max_duration_seconds,
            "max_goal": self.max_goal,
            "team_list": [
                {"players_number": team.players_number, "name": team.name, "color": list(team.color)}
                for team in self.team_list
            ],
        }


@dataclass
class GameModeTeamRange:
    players_number: tuple[int, int]
    name: str = None
    color: Color = None

    def _resolve_identity(self, team_index: int) -> tuple[str, Color]:
        if self.name is not None and self.color is not None:
            return self.name, self.color

        name, color = DEFAULT_TEAM_PALETTE[team_index]
        return self.name or name, self.color or color

    def sample(self, rng: np.random.Generator, team_index: int) -> GameModeTeam:
        name, color = self._resolve_identity(team_index)
        return GameModeTeam(
            players_number=int(rng.integers(self.players_number[0], self.players_number[1] + 1)),
            name=name,
            color=color,
        )

    def max_team(self, team_index: int) -> GameModeTeam:
        name, color = self._resolve_identity(team_index)
        return GameModeTeam(players_number=self.players_number[1], name=name, color=color)


# Same shape as GameMode, but every field is a (min, max) range instead of a concrete
# value — a fixed value is just a degenerate range like (5, 5). Cubeball samples a fresh
# GameMode from this every episode (see GameMode.sample below / Cubeball.reset).
@dataclass
class GameModeRange:
    level_size: tuple[Vector3, Vector3]
    goal_size: tuple[Vector3, Vector3]
    ball_number: tuple[int, int]
    obstacle_number: tuple[int, int]
    max_duration_seconds: tuple[float, float]
    max_goal: tuple[int, int]
    team_list: list[GameModeTeamRange]

    def sample(self, rng: np.random.Generator) -> GameMode:
        return GameMode(
            level_size=tuple(float(rng.uniform(low, high)) for low, high in zip(*self.level_size)),
            goal_size=tuple(float(rng.uniform(low, high)) for low, high in zip(*self.goal_size)),
            ball_number=int(rng.integers(self.ball_number[0], self.ball_number[1] + 1)),
            obstacle_number=int(rng.integers(self.obstacle_number[0], self.obstacle_number[1] + 1)),
            max_duration_seconds=float(rng.uniform(*self.max_duration_seconds)),
            max_goal=int(rng.integers(self.max_goal[0], self.max_goal[1] + 1)),
            team_list=[team.sample(rng, team_index) for team_index, team in enumerate(self.team_list)],
        )

    # Configuration used purely to discover the observation/action space of every
    # possible_agent up front (see Cubeball.__init__ / CubeballConnection.get_spaces):
    # every team at its largest roster, so no agent_id is ever missing from the reply.
    def max_game_mode(self) -> GameMode:
        return GameMode(
            level_size=self.level_size[1],
            goal_size=self.goal_size[1],
            ball_number=self.ball_number[1],
            obstacle_number=self.obstacle_number[1],
            max_duration_seconds=self.max_duration_seconds[1],
            max_goal=self.max_goal[1],
            team_list=[team.max_team(team_index) for team_index, team in enumerate(self.team_list)],
        )
