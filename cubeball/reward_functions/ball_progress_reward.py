import numpy as np

from cubeball.reward_functions.reward_function import (
    BALL_ENTITY_TYPE,
    RewardFunction,
    agent_team_name_from_entities,
    broadcast_team_rewards_to_agents,
    goal_reward_by_team_name,
)


class BallProgress(RewardFunction):
    """
    Step reward = -(closest ball distance to opponent goal / level_size_norm) * (continuous_weight / max_steps).

    max_steps is passed explicitly so the reward scale stays fixed regardless of the actual
    episode length. If max_steps matched info["max_steps"] and episodes ended early, the
    cumulative reward would be much smaller than continuous_weight.
    """

    def __init__(self, continuous_weight: float = 0.5, max_steps: int = 200):
        self._step_scale: float = continuous_weight / max_steps
        self._goal_position_by_team_name: dict = {}
        self._level_size_norm: float = 1.0

    def reset(self, info: dict) -> None:
        self._level_size_norm = float(np.linalg.norm(info["level_size"]))
        self._goal_position_by_team_name = {
            team_name: np.array(position, dtype=np.float64)
            for team_name, position in info["goals"].items()
        }

    def compute_rewards(self, info: dict) -> dict:
        # if info["goal_events"]:
        #     team_rewards = goal_reward_by_team_name(info)
        # else:
        team_rewards = {
            team_name: self._compute_team_step_reward(info["entities"], team_name)
            for team_name in self._goal_position_by_team_name
        }

        agent_team_name = agent_team_name_from_entities(info["entities"])
        return broadcast_team_rewards_to_agents(team_rewards, agent_team_name)

    def _compute_team_step_reward(self, entities: list, team_name: str) -> float:
        ball_positions = [
            np.array(entity["position"], dtype=np.float64)
            for entity in entities
            if entity["entity_type"] == BALL_ENTITY_TYPE
        ]
        if not ball_positions:
            return 0.0

        opponent_team_name = next(n for n in self._goal_position_by_team_name if n != team_name)
        opponent_goal_position = self._goal_position_by_team_name[opponent_team_name]

        closest_ball_distance = min(
            np.linalg.norm(ball_position - opponent_goal_position)
            for ball_position in ball_positions
        )

        return -min(closest_ball_distance / self._level_size_norm, 1.0) * self._step_scale
