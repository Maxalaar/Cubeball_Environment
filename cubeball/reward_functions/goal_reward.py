from cubeball.reward_functions.reward_function import (
    RewardFunction,
    agent_team_name_from_entities,
    broadcast_team_rewards_to_agents,
    goal_reward_by_team_name,
)


class GoalReward(RewardFunction):
    """Sparse ±1 per goal event, 0 otherwise. Mirrors the reward that used to be computed
    in Godot (CuboidAIController._on_goal_scored)."""

    def reset(self, _info: dict) -> None:
        pass

    def compute_rewards(self, info: dict) -> dict:
        agent_team_name = agent_team_name_from_entities(info["entities"])
        team_rewards = goal_reward_by_team_name(info)
        return broadcast_team_rewards_to_agents(team_rewards, agent_team_name)
