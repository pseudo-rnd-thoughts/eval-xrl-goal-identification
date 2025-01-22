import os
from copy import deepcopy
from typing import Any, SupportsFloat

import ale_py
import flax
import gymnasium as gym
import jax
import numpy as np
from gymnasium.core import ObsType, WrapperActType, WrapperObsType

gym.register_envs(ale_py)

# the first 290+ (raw) actions in the environment are "dead" they don't do anything, therefore, we can skip them through no-op them
DEAD_INITIAL_ACTIONS = 65


class LossOfLifeReward(gym.Wrapper):

    def __init__(self, env: gym.Env, reward: callable):
        assert isinstance(env, gym.wrappers.AtariPreprocessing)
        super().__init__(env)
        self.reward: callable = reward

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, self.reward(terminated), terminated, truncated, info


class EnergyPillReward(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.timesteps_since_energypill = 0
        self.energypills_eaten = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        self.energypills_eaten, self.timesteps_since_energypill = 0, 0
        return super().reset(seed=seed)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.energypills_eaten > 0:
            self.timesteps_since_energypill += 1
        if reward == 50:
            self.energypills_eaten += 1
            self.timesteps_since_energypill = 0
        # energy pill (50), ghosts eaten (200, 400, 800, 1600)
        # There are cases where the agent can eat a dot at the same time as a ghost giving +10 points
        reward = int(reward in (50, 200, 210, 400, 410, 800, 810, 1600, 1610))

        truncated = (self.energypills_eaten == 4 and self.timesteps_since_energypill > 500) or truncated
        return obs, reward, terminated, truncated, info


class ExpertStart(gym.Wrapper):
    def __init__(self, env,
                 expert_agent_filename="models/MsPacman-v5_dqn_train.cleanrl_model",
                 min_expert_actions: int = 0,
                 max_expert_actions: int = 140,
                 expert_epsilon: float = 0.01):
        super().__init__(env)

        from explanation_mechanisms.dqn_train import ImpalaQNetwork

        self.q_network = ImpalaQNetwork(action_dim=self.env.action_space.n)
        filename = os.path.join(os.path.dirname(__file__), "..", expert_agent_filename)
        with open(filename, "rb") as f:
            self.q_params = self.q_network.init(jax.random.PRNGKey(0), np.zeros((1, 4, 84, 84)))
            self.q_params = flax.serialization.from_bytes(self.q_params, f.read())

        self.min_expert_actions = min_expert_actions
        self.max_expert_actions = max_expert_actions
        self.expert_epsilon = expert_epsilon

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        num_expert_actions = self.np_random.integers(low=self.min_expert_actions, high=self.max_expert_actions)
        for _ in range(num_expert_actions):
            if self.np_random.random() < self.expert_epsilon:
                action = self.action_space.sample()
            else:
                action = jax.device_get(self.q_network.apply(self.q_params, np.expand_dims(obs, axis=0))).argmax(axis=-1)[0]
            obs, _, terminated, _, info = self.env.step(action)
            if terminated:
                obs, info = self.reset()

        return obs, info


class SkipDeadActions(gym.Wrapper):

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(DEAD_INITIAL_ACTIONS):
            obs, _, _, _, info = self.env.step(0)
        return obs, info


class SkipDeathAnimation(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.ale = env.unwrapped.ale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.ale.getRAM()[39] == 73:
            terminated = True
        return obs, reward, terminated, truncated, info


mspacman_kwargs = gym.spec("ALE/MsPacman-v5").kwargs
mspacman_kwargs["frameskip"] = 1
gym.register(
    id="MsPacman-v5", entry_point="ale_py.env:AtariEnv",
    kwargs=deepcopy(mspacman_kwargs),
    additional_wrappers=(
        gym.wrappers.AtariPreprocessing.wrapper_spec(terminal_on_life_loss=True, noop_max=0),
        gym.wrappers.FrameStackObservation.wrapper_spec(stack_size=4),
        SkipDeadActions.wrapper_spec()
    )
)
gym.register(
    id="MsPacman-Dots-v2", entry_point="ale_py.env:AtariEnv",
    kwargs=deepcopy(mspacman_kwargs),
    additional_wrappers=(
        # dots (10), there are cases where the agent can eat a ghost or cherry at the same time as a dot giving +X values that need to be accounted for.
        gym.wrappers.TransformReward.wrapper_spec(func=lambda reward: int(reward in (10, 110, 210, 410, 810, 1610, 510, 710))),
        gym.wrappers.AtariPreprocessing.wrapper_spec(terminal_on_life_loss=True, noop_max=0),
        gym.wrappers.FrameStackObservation.wrapper_spec(stack_size=4),
        SkipDeadActions.wrapper_spec(),
    )
)

gym.register(
    id="MsPacman-EnergyPill-v2", entry_point="ale_py.env:AtariEnv",
    kwargs=deepcopy(mspacman_kwargs),
    additional_wrappers=(
        # energy pill (50), ghosts eaten (200, 400, 800, 1600) - there are cases where the agent can eat a dot at the same time as a ghost giving +10 points
        gym.wrappers.TransformReward.wrapper_spec(func=lambda reward: int(reward in (50, 200, 210, 400, 410, 800, 810, 1600, 1610))),
        gym.wrappers.AtariPreprocessing.wrapper_spec(terminal_on_life_loss=True, noop_max=0),
        gym.wrappers.FrameStackObservation.wrapper_spec(stack_size=4),
        SkipDeadActions.wrapper_spec(),
    )
)

gym.register(
    id="MsPacman-Survival-v2", entry_point="ale_py.env:AtariEnv",
    kwargs=deepcopy(mspacman_kwargs),
    additional_wrappers=(
        gym.wrappers.AtariPreprocessing.wrapper_spec(terminal_on_life_loss=True, noop_max=0),
        LossOfLifeReward.wrapper_spec(reward=lambda terminated: .5 * int(not terminated)),
        gym.wrappers.FrameStackObservation.wrapper_spec(stack_size=4),
        SkipDeadActions.wrapper_spec(),
    )
)

gym.register(
    id="MsPacman-LoseALife-v2", entry_point="ale_py.env:AtariEnv",
    kwargs=deepcopy(mspacman_kwargs),
    additional_wrappers=(
        gym.wrappers.AtariPreprocessing.wrapper_spec(terminal_on_life_loss=True, noop_max=0),
        LossOfLifeReward.wrapper_spec(reward=lambda terminated: -.5 * int(not terminated)),
        gym.wrappers.FrameStackObservation.wrapper_spec(stack_size=4),
        SkipDeadActions.wrapper_spec(),
        ExpertStart.wrapper_spec(),
    )
)


def make_env(env_id, seed, env_idx, capture_video, video_folder):
    def thunk():
        if capture_video and env_idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: True)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env.action_space.seed(seed)
        return env

    return thunk
