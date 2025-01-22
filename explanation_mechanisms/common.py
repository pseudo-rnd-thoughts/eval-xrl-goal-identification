from collections import namedtuple

import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as onp

from explanation_mechanisms.mspacman_envs import make_env


def evaluate(env_id: str, seed: int, video_folder: str, epsilon, model, model_filename, capture_video=True, num_episodes: int = 10):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, seed, env_idx=0, capture_video=capture_video, video_folder=video_folder)])

    q_network = model(action_dim=envs.single_action_space.n)
    with open(model_filename, "rb") as f:
        q_params = q_network.init(jax.random.PRNGKey(seed), envs.observation_space.sample())
        q_params = flax.serialization.from_bytes(q_params, f.read())

    obs, _ = envs.reset(seed=seed)
    episodic_returns = []
    while len(episodic_returns) < num_episodes:
        if onp.random.random() < epsilon:
            actions = onp.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network.apply(q_params, obs)
            actions = jnp.argmax(q_values, axis=-1)
            actions = jax.device_get(actions)

        obs, _, _, _, infos = envs.step(actions)
        if "episode" in infos:
            print(f"eval_episode={len(episodic_returns)}, episodic_return={infos['episode']['r'][0]}")
            episodic_returns.append(infos["episode"]["r"][0])
    envs.close()
    return episodic_returns


ReplayData = namedtuple("ReplayData", ["observations", "actions", "next_observations", "rewards", "terminations"])


class ReplayBuffer:

    def __init__(self, buffer_size: int, obs_space, action_space):
        self.buffer_size = buffer_size

        self.obs_buffer = onp.zeros((self.buffer_size, *obs_space.shape), dtype=obs_space.dtype)
        self.action_buffer = onp.zeros((self.buffer_size, *action_space.shape), dtype=action_space.dtype)
        self.next_obs_buffer = onp.zeros((self.buffer_size, *obs_space.shape), dtype=obs_space.dtype)
        self.reward_buffer = onp.zeros((self.buffer_size, 1), dtype=onp.float32)
        self.terminated_buffer = onp.zeros((self.buffer_size, 1), dtype=onp.bool_)

        self.idx: int = 0
        self.full: bool = False

    def add(self, obs, action, next_obs, reward, terminated):
        self.obs_buffer[self.idx] = obs
        self.action_buffer[self.idx] = action
        self.next_obs_buffer[self.idx] = next_obs
        self.reward_buffer[self.idx] = reward
        self.terminated_buffer[self.idx] = terminated

        self.idx += 1
        if self.idx == self.buffer_size:
            self.full, self.idx = True, 0

    def sample(self, batch_size: int) -> ReplayData:
        if self.full:
            sample_idx = onp.random.randint(0, self.buffer_size, size=batch_size)
        else:
            sample_idx = onp.random.randint(0, self.idx, size=batch_size)

        return ReplayData(
            self.obs_buffer[sample_idx],
            self.action_buffer[sample_idx],
            self.next_obs_buffer[sample_idx],
            self.reward_buffer[sample_idx],
            self.terminated_buffer[sample_idx]
        )
