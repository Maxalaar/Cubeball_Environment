from pathlib import Path
from typing import Optional
import gymnasium as gym
import numpy as np
from ray.rllib import MultiAgentEnv

from cubeball.game_mode import GameModeRange
from cubeball.connection import CubeballConnection, get_free_port
from cubeball.reward_functions import RewardFunction

GAME_EXECUTABLE_PATH = str(Path(__file__).parent.parent / "cubeball_godot" / "Cubeball.x86_64")


def build_observation_space(schema: dict) -> gym.spaces.Dict:
    spaces_by_key = {}

    for key, value in schema.items():
        if value["space"] == "continuous":
            size = value["size"]
            shape = tuple(size) if isinstance(size, (list, tuple)) else (size,)
            spaces_by_key[key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        elif value["space"] == "binary":
            spaces_by_key[key] = gym.spaces.MultiBinary(value["size"])
        elif value["space"] == "discrete":
            spaces_by_key[key] = gym.spaces.Discrete(value["size"])
        else:
            raise ValueError(f"Unsupported observation space kind: {value['space']!r}")

    return gym.spaces.Dict(spaces_by_key)


def build_action_space(schema: dict) -> gym.spaces.Dict:
    spaces_by_key = {}

    for key, value in schema.items():
        if value["action_type"] == "discrete":
            spaces_by_key[key] = gym.spaces.Discrete(value["size"])
        elif value["action_type"] == "continuous":
            spaces_by_key[key] = gym.spaces.Box(low=-1.0, high=1.0, shape=(value["size"],))
        else:
            raise ValueError(f"Unsupported action type: {value['action_type']!r}")

    return gym.spaces.Dict(spaces_by_key)


class Cubeball(MultiAgentEnv):
    def __init__(self, environment_configuration: Optional[dict] = None):
        super().__init__()

        if environment_configuration.get('render_mode', None) is not None:
            environment_configuration["show_window"] = True
            environment_configuration["speedup"] = 1.0

        self.game_mode_range: GameModeRange = environment_configuration["game_mode_range"]
        reward_function_class = environment_configuration["reward_function"]
        self.reward_function: RewardFunction = reward_function_class()
        self._rng = np.random.default_rng(environment_configuration.get("seed"))

        self.connection = CubeballConnection(
            env_path=GAME_EXECUTABLE_PATH,
            port=get_free_port(),
            show_window=environment_configuration["show_window"],
            action_repeat=environment_configuration["action_repeat"],
            speedup=environment_configuration["speedup"],
            debug_logs=environment_configuration.get("debug_logs", False),
            observation_mode=environment_configuration.get("observation_mode", "raycast"),
            disable_goal_animation=environment_configuration.get("disable_goal_animation", True),
            disable_ui=environment_configuration.get("disable_ui", True),
            disable_goal_nets=environment_configuration.get("disable_goal_nets", True),
            disable_cameras=environment_configuration.get("disable_cameras", True),
            disable_environment=environment_configuration.get("disable_environment", True),
            display_fps=environment_configuration.get("display_fps", False),
        )

        spaces_reply = self.connection.get_spaces(self.game_mode_range.max_game_mode().to_config())

        self.possible_agents = sorted(spaces_reply["observation_space"].keys())
        self.agents = list(self.possible_agents)
        self.agent_policy_names: dict = spaces_reply["agent_policy_names"]

        self.observation_spaces = {
            agent_id: build_observation_space(schema)
            for agent_id, schema in spaces_reply["observation_space"].items()
        }
        self.action_spaces = {
            agent_id: build_action_space(schema) for agent_id, schema in spaces_reply["action_space"].items()
        }

        self._current_game_mode = self.game_mode_range.sample(self._rng)
        self._pending_reset_reply = self.connection.reset(self._current_game_mode.to_config())

        self.use_real_godot_done: float = environment_configuration.get('use_real_godot_done', True)
        self.reward_scale_factor: float = environment_configuration.get('reward_scale_factor', 1.0)
        self.current_step: Optional[int] = None
        self.max_step: Optional[int] = environment_configuration.get("max_step", None)

        self.observation_space = gym.spaces.Dict(self.observation_spaces)
        self.action_space = gym.spaces.Dict(self.action_spaces)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._pending_reset_reply is not None:
            reply = self._pending_reset_reply
            self._pending_reset_reply = None
        else:
            self._current_game_mode = self.game_mode_range.sample(self._rng)
            reply = self.connection.reset(self._current_game_mode.to_config())

        self.agents = list(reply["observation"].keys())
        self.current_step = 0

        self.reward_function.reset(reply["info"])

        observation = self.process_observations(reply["observation"])
        information = {agent_id: {} for agent_id in self.agents}
        return observation, information

    def step(self, action_dict):
        self.current_step += 1

        reply = self.connection.step(self.process_actions(action_dict))
        observation = self.process_observations(reply["observation"])
        reward = {
            agent_id: agent_reward * self.reward_scale_factor
            for agent_id, agent_reward in self.reward_function.compute_rewards(reply["info"]).items()
        }
        done = self.process_dones(reply["done"])
        truncated = self.process_truncates()
        information = {agent_id: {} for agent_id in self.agents}

        if self.max_step is not None and self.current_step >= self.max_step:
            done = self.dones_for_all()

        return observation, reward, done, truncated, information

    def close(self):
        self.connection.close()

    def process_observations(self, observations: dict) -> dict:
        return {
            agent_id: self._cast_observation(self.observation_spaces[agent_id], observation)
            for agent_id, observation in observations.items()
        }

    @staticmethod
    def _cast_observation(space: gym.spaces.Dict, observation: dict) -> dict:
        casted = {}
        for key, sub_space in space.spaces.items():
            if isinstance(sub_space, gym.spaces.Box):
                casted[key] = np.array(observation[key], dtype=sub_space.dtype)
            elif isinstance(sub_space, gym.spaces.MultiBinary):
                casted[key] = np.array(observation[key], dtype=np.int8)
            elif isinstance(sub_space, gym.spaces.Discrete):
                casted[key] = int(observation[key])
            else:
                raise ValueError(f"Unsupported observation space kind: {sub_space!r}")
        return casted

    def process_actions(self, action_dict: dict) -> dict:
        return {
            agent_id: self._cast_action(self.action_spaces[agent_id], action)
            for agent_id, action in action_dict.items()
            if agent_id in self.agents
        }

    @staticmethod
    def _cast_action(space: gym.spaces.Dict, action: dict) -> dict:
        casted = {}
        for key, sub_space in space.spaces.items():
            if isinstance(sub_space, gym.spaces.Box):
                casted[key] = np.asarray(action[key]).tolist()
            elif isinstance(sub_space, gym.spaces.Discrete):
                casted[key] = int(action[key])
            else:
                raise ValueError(f"Unsupported action space kind: {sub_space!r}")
        return casted

    def process_dones(self, dones: dict) -> dict:
        if not self.use_real_godot_done:
            dones = {agent_id: False for agent_id in dones}
        dones_dict = dict(dones)
        dones_dict['__all__'] = all(dones.values())
        return dones_dict

    def dones_for_all(self):
        dones_dict = {agent_id: True for agent_id in self.agents}
        dones_dict['__all__'] = True
        return dones_dict

    def process_truncates(self) -> dict:
        return {agent_id: False for agent_id in self.agents}
