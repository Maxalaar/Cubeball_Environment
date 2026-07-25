from abc import ABC, abstractmethod

import numpy as np

CUBOID_ENTITY_TYPE = "CUBOID"
BALL_ENTITY_TYPE = "BALL"


class RewardFunction(ABC):
    """Computes a per-agent reward from the training info bundle Godot sends every reply
    (see PythonSynchronizer._get_training_info in the Godot repo). Teams are identified by
    name throughout (Team.name on the Godot side) — never by a positional index."""

    @abstractmethod
    def reset(self, info: dict) -> None:
        """Called once per episode, right after the reset reply arrives."""

    @abstractmethod
    def compute_rewards(self, info: dict) -> dict:
        """Called once per step. Returns {agent_id: reward}, each within [-1, 1]."""


def agent_team_name_from_entities(entities: list) -> dict:
    return {
        entity["agent_id"]: entity["team_name"]
        for entity in entities
        if entity["entity_type"] == CUBOID_ENTITY_TYPE
    }


def goal_reward_by_team_name(info: dict) -> dict:
    team_names = set(info["goals"].keys())
    goal_reward_by_team_name = {team_name: 0.0 for team_name in team_names}

    for event in info["goal_events"]:
        receiving_team_name = event["receiving_team_name"]
        for team_name in team_names:
            goal_reward_by_team_name[team_name] += -1.0 if team_name == receiving_team_name else 1.0

    return {
        team_name: float(np.clip(reward, -1.0, 1.0))
        for team_name, reward in goal_reward_by_team_name.items()
    }


def broadcast_team_rewards_to_agents(team_reward_by_team_name: dict, agent_team_name: dict) -> dict:
    return {
        agent_id: team_reward_by_team_name[team_name]
        for agent_id, team_name in agent_team_name.items()
    }
