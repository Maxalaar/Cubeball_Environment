import socket
from typing import Optional
import gymnasium as gym
import numpy as np
from ray.rllib import MultiAgentEnv
from gymnasium.spaces.utils import flatten_space

try:
    from godot_rl.core.godot_env import GodotEnv
    from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv
except ImportError:
    print('Godot RL is not installed.')


def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

# https://github.com/edbeeching/godot_rl_agents_examples
# https://github.com/edbeeching/godot_rl_agents/blob/main/docs/TRAINING_MULTIPLE_POLICIES.md
class Cubeball(MultiAgentEnv):
    def __init__(self, environment_configuration: Optional[dict] = None):
        super().__init__()

        if environment_configuration.get('render_mode', None) is not None:
            environment_configuration["show_window"] = True
            environment_configuration["speedup"] = 1.0

        self.environment: GodotEnv = GodotEnv(
            env_path="cubeball_godot/Cubeball.x86_64",
            # env_path="super_slime_volley_godot/Super_Slime_Volley.x86_64",
            port=get_free_port(),
            show_window=environment_configuration["show_window"],
            action_repeat=environment_configuration["action_repeat"],
            speedup=environment_configuration["speedup"],
        )
        print('Policy names: ' + str(self.environment.agent_policy_names))

        self.agents = []
        self.possible_agents = []
        self.real_observation_spaces = {}
        self.observation_spaces = {}
        self.action_spaces = {}

        for i, agent_policy_name in enumerate(self.environment.agent_policy_names):
            self.agents.append(agent_policy_name)
            self.possible_agents.append(agent_policy_name)

            self.real_observation_spaces[agent_policy_name] = self.environment.observation_spaces[i]
            self.observation_spaces[agent_policy_name] = flatten_space(self.environment.observation_spaces[i])

            self.action_spaces[agent_policy_name] = self.environment.action_spaces[i]

        self.use_real_godot_done: float = environment_configuration.get('use_real_godot_done', True)
        self.reward_scale_factor: float = environment_configuration.get('reward_scale_factor', 1.0)
        self.current_step: Optional[int] = None
        self.max_step: Optional[int] = environment_configuration.get("max_step", None)

        self.observation_space = gym.spaces.Dict(self.observation_spaces)
        self.action_space = gym.spaces.Dict(self.action_spaces)

    def reset(self, seed=None, options=None):
        observation, information = self.environment.reset(seed=seed)
        self.current_step = 0
        observation = self.process_observations(observation)
        information = self.process_information(information)
        return observation, information

    def step(self, action_dict):
        self.current_step += 1

        actions = self.process_actions(action_dict)
        observation, reward, done, truncated, information = self.environment.step(actions, order_ij=True)
        observation = self.process_observations(observation)
        reward = self.process_rewards(reward)
        done = self.process_dones(done)
        truncated = self.process_truncates(truncated)
        information = self.process_information(information)

        if self.max_step is not None and self.current_step >= self.max_step:
            done = self.dones_for_all()

        return observation, reward, done, truncated, information

    def close(self):
        self.environment.close()

    def process_observations(self, observations):
        new_observations = []

        for i, observation in enumerate(observations):
            agent_name = self.possible_agents[i]
            new_observation = gym.spaces.utils.flatten(self.real_observation_spaces[agent_name], observation)
            new_observations.append(new_observation)

        return dict(zip(self.possible_agents, new_observations))

    def process_information(self, information):
        return dict(zip(self.possible_agents, information))

    def process_actions(self, action_dict):
        new_actions = []

        for agent_name in self.agents:
            if agent_name in action_dict.keys():
                new_action = list(action_dict[agent_name].values())
                new_actions.append(new_action)

        return new_actions

    def process_rewards(self, rewards):
        rewards = np.array(rewards) * self.reward_scale_factor
        return dict(zip(self.possible_agents, rewards))

    def process_dones(self, dones):
        if not self.use_real_godot_done:
            dones = [False for _ in range(len(self.possible_agents))]
        return dict(zip(self.possible_agents, dones))

    def dones_for_all(self):
        dones = [True for _ in range(len(self.possible_agents))]
        return dict(zip(self.possible_agents, dones))

    def process_truncates(self, truncates):
        truncates = [False for _ in range(len(self.possible_agents))]
        return dict(zip(self.possible_agents, truncates))
