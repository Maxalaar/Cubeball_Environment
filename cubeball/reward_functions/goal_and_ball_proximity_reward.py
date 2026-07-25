import numpy as np

from cubeball.reward_functions.reward_function import (
    BALL_ENTITY_TYPE,
    RewardFunction,
    agent_team_name_from_entities,
    broadcast_team_rewards_to_agents,
    goal_reward_by_team_name,
)


class GoalAndBallProximity(RewardFunction):
    """Sparse goal reward, plus dense potential-based shaping on ball-to-goal proximity on
    steps without a goal event (the closest ball to each goal, since several can be in
    play). The shaping term is a delta of proximity between two steps, not an absolute
    snapshot, so it can't be farmed by camping near the opponent's goal and its sum over an
    episode stays bounded regardless of episode length (Ng et al. 1999)."""

    def __init__(self, shaping_cap: float = 0.2):
        self.shaping_cap = shaping_cap
        self._goal_position_by_team_name: dict = {}
        self._level_size_norm: float = 1.0
        self._previous_team_potential: dict = {}

    def reset(self, info: dict) -> None:
        self._level_size_norm = float(np.linalg.norm(info["level_size"]))
        self._goal_position_by_team_name = {
            team_name: np.array(position, dtype=np.float64) for team_name, position in info["goals"].items()
        }
        self._previous_team_potential = {
            team_name: self._compute_team_potential(info["entities"], team_name)
            for team_name in self._goal_position_by_team_name
        }

    def compute_rewards(self, info: dict) -> dict:
        current_team_potential = {
            team_name: self._compute_team_potential(info["entities"], team_name)
            for team_name in self._goal_position_by_team_name
        }

        if info["goal_events"]:
            # Sparse reward dominates any step a goal happened on — shaping is skipped
            # entirely, since the ball sitting in the net would otherwise double-count the
            # same signal the goal event already represents.
            team_rewards = goal_reward_by_team_name(info)
        else:
            team_rewards = {
                team_name: float(
                    np.clip(
                        current_team_potential[team_name] - self._previous_team_potential[team_name],
                        -self.shaping_cap,
                        self.shaping_cap,
                    )
                )
                for team_name in self._goal_position_by_team_name
            }

        self._previous_team_potential = current_team_potential

        agent_team_name = agent_team_name_from_entities(info["entities"])
        return broadcast_team_rewards_to_agents(team_rewards, agent_team_name)

    def _compute_team_potential(self, entities: list, team_name: str) -> float:
        ball_positions = [
            np.array(entity["position"], dtype=np.float64)
            for entity in entities
            if entity["entity_type"] == BALL_ENTITY_TYPE
        ]
        if not ball_positions:
            return 0.0

        # Exactly 2 teams/goals exist on the field (see Level.build_level in the Godot
        # repo, which always spawns 2 goals), so the opponent is simply whichever other
        # name is in the goal dict.
        opponent_team_name = next(name for name in self._goal_position_by_team_name if name != team_name)
        own_goal_position = self._goal_position_by_team_name[team_name]
        opponent_goal_position = self._goal_position_by_team_name[opponent_team_name]

        # Multiple balls can be in play: only the single most threatening/advanced ball
        # matters for each term.
        closest_ball_distance_to_own_goal = min(
            np.linalg.norm(ball_position - own_goal_position) for ball_position in ball_positions
        )
        closest_ball_distance_to_opponent_goal = min(
            np.linalg.norm(ball_position - opponent_goal_position) for ball_position in ball_positions
        )

        # Same normalization PhysicsEntity.get_observation_informations already uses for
        # the policy observation (physics_entity.gd), reused here for consistency.
        own_goal_distance_normalized = min(closest_ball_distance_to_own_goal / self._level_size_norm, 1.0)
        opponent_goal_distance_normalized = min(
            closest_ball_distance_to_opponent_goal / self._level_size_norm, 1.0
        )

        return own_goal_distance_normalized - opponent_goal_distance_normalized
