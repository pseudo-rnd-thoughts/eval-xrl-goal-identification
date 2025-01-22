"""This code is original based off https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/qdagger_dqn_atari_jax_impalacnn.py"""

import os

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import random
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Sequence

import chex
import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from rich.progress import track
from torch.utils.tensorboard import SummaryWriter

from explanation_mechanisms.common import evaluate, ReplayBuffer
from explanation_mechanisms.dqn_train import ImpalaQNetwork as TeacherModel, ConvSequence
from explanation_mechanisms.mspacman_envs import make_env


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "TrdUserSurvey"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

    teacher_eval_episodes: int = 10
    """the number of episodes to run the teacher policy evaluate"""
    teacher_steps: int = 500_000
    """the number of steps to run the teacher policy to generate the replay buffer"""
    offline_steps: int = 500_000
    """the number of steps to update the student policy with the teacher's replay buffer"""
    temperature: float = 1.0
    """the temperature parameter for qdagger"""

    reward_vector_size: int = 41
    """The size of the reward vector prediction."""


# ALGO LOGIC: initialize agent here:
class NatureQNetwork(nn.Module):
    action_dim: int
    reward_vector_size: int

    def __call__(self, x: jnp.ndarray):
        return jnp.sum(self.decomposed_q_value(x), axis=-1)

    @nn.compact
    def decomposed_q_value(self, x: jnp.ndarray):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.reward_vector_size)(x)
        return jnp.reshape(x, (-1, self.action_dim, self.reward_vector_size))


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ImpalaQNetwork(nn.Module):
    action_dim: int
    reward_vector_size: int
    channelss: Sequence[int] = (16, 32, 32)

    def __call__(self, x: jnp.ndarray):
        return jnp.sum(self.decomposed_q_value(x), axis=-1)

    @nn.compact
    def decomposed_q_value(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / 255.0
        for channels in self.channelss:
            x = ConvSequence(channels)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.reward_vector_size)(x)
        return jnp.reshape(x, (-1, self.action_dim, self.reward_vector_size))


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__rv{args.reward_vector_size}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, env_idx=0, capture_video=False, video_folder="")])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = ImpalaQNetwork(action_dim=envs.single_action_space.n, reward_vector_size=args.reward_vector_size)
    q_network.apply = jax.jit(q_network.apply, static_argnames=("method",))

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, envs.observation_space.sample()),
        target_params=q_network.init(q_key, envs.observation_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    # QDAGGER LOGIC:
    teacher_model_path = f"models/{args.env_id}_dqn_train.cleanrl_model"
    teacher_model = TeacherModel(action_dim=envs.single_action_space.n)
    with open(teacher_model_path, "rb") as f:
        teacher_params = teacher_model.init(jax.random.PRNGKey(args.seed), envs.observation_space.sample())
        teacher_params = flax.serialization.from_bytes(teacher_params, f.read())
    teacher_model.apply = jax.jit(teacher_model.apply)

    # Helper variables
    batch_size = args.batch_size
    num_actions = envs.single_action_space.n
    reward_vector_size = args.reward_vector_size

    # evaluate the teacher model
    teacher_episodic_returns = evaluate(
        args.env_id,
        args.seed,
        epsilon=args.end_e,
        model=TeacherModel,
        model_filename=teacher_model_path,
        video_folder="",
        capture_video=False,
        num_episodes=args.teacher_eval_episodes
    )
    for idx, episode_return in enumerate(teacher_episodic_returns):
        writer.add_scalar(f"teacher/episodic_return", episode_return, idx)

    # collect teacher data for args.teacher_steps
    # we assume we don't have access to the teacher's replay buffer
    # see Fig. A.19 in Agarwal et al. 2022 for more detail
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
    )

    start_time = time.time()
    # print(f'Started filling: {start_time}')
    obs, _ = envs.reset(seed=args.seed)
    autoreset = np.zeros((1,), dtype=np.bool_)
    for global_step in track(range(args.teacher_steps), description="filling teacher's replay buffer"):
        # linear_schedule(args.start_e, args.end_e, args.teacher_steps, global_step)
        if random.random() < args.end_e:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = teacher_model.apply(teacher_params, obs)
            actions = jax.device_get(q_values.argmax(axis=-1))
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        if not np.any(autoreset):
            rb.add(obs, actions, next_obs, rewards, terminated)
        obs = next_obs
        autoreset = np.logical_or(terminated, truncated)

    end_time = time.time()
    print(f'Teacher replay buffer fill time: {end_time - start_time:.2f} seconds')

    @jax.vmap
    def kl_divergence_with_logits(target_logits, prediction_logits):
        """Implementation of on-policy distillation loss."""
        out = -nn.softmax(target_logits) * (nn.log_softmax(prediction_logits) - nn.log_softmax(target_logits))
        return jnp.sum(out)

    @jax.jit
    def update(q_state, train_obs, train_actions, train_next_obs, train_rewards, train_terminated, distill_coeff):
        # Temporal reward decomposition loss function
        q_next_target = q_network.apply(q_state.target_params, train_next_obs, method=ImpalaQNetwork.decomposed_q_value)
        chex.assert_shape(q_next_target, (batch_size, num_actions, reward_vector_size))
        q_next_target_value = q_next_target[jnp.arange(args.batch_size), jnp.argmax(jnp.sum(q_next_target, axis=-1), axis=-1)]
        chex.assert_shape(q_next_target_value, (batch_size, reward_vector_size))

        discounted_q_next_target = jnp.expand_dims(1 - train_terminated.squeeze(), axis=1) * args.gamma * q_next_target_value
        chex.assert_shape(discounted_q_next_target, (batch_size, reward_vector_size))
        rolled_q_next_target = jnp.roll(discounted_q_next_target, shift=1, axis=1)
        chex.assert_shape(rolled_q_next_target, (batch_size, reward_vector_size))
        next_q_value = rolled_q_next_target.at[:, -1].add(rolled_q_next_target[:, 0]).at[:, 0].set(train_rewards.squeeze())
        chex.assert_shape(next_q_value, (batch_size, reward_vector_size))

        teacher_q_values = teacher_model.apply(teacher_params, train_obs)
        chex.assert_shape(teacher_q_values, (batch_size, num_actions))

        def qdagger_trd_loss(params, td_target, teacher_q_values):
            student_q_values = q_network.apply(params, train_obs, method=ImpalaQNetwork.decomposed_q_value)
            chex.assert_shape(student_q_values, (batch_size, num_actions, reward_vector_size))

            # td loss
            q_pred = student_q_values[jnp.arange(batch_size), train_actions.squeeze()]
            chex.assert_shape(q_pred, (batch_size, reward_vector_size))
            q_loss = jnp.mean(jnp.square(q_pred - td_target))
            chex.assert_shape(q_loss, ())

            # distil loss
            teacher_q_values = teacher_q_values / args.temperature
            student_q_values = jnp.sum(student_q_values, axis=-1) / args.temperature
            chex.assert_shape(teacher_q_values, (batch_size, num_actions))
            chex.assert_shape(student_q_values, (batch_size, num_actions))
            policy_divergence = kl_divergence_with_logits(teacher_q_values, student_q_values)
            chex.assert_shape(policy_divergence, (batch_size,))
            distill_loss = distill_coeff * jnp.mean(policy_divergence)
            chex.assert_shape(distill_loss, ())

            overall_loss = q_loss + distill_loss
            chex.assert_shape(overall_loss, ())

            # purely to show that the student q-value convergences to the teacher's
            teacher_student_error = jnp.mean(jnp.square(student_q_values - teacher_q_values))

            return overall_loss, (q_loss, q_pred, distill_loss, teacher_student_error)

        (loss_value, (q_loss, q_pred, distill_loss, teacher_student_error)), grads = jax.value_and_grad(qdagger_trd_loss, has_aux=True)(
            q_state.params, next_q_value, teacher_q_values
        )
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_loss, q_pred, distill_loss, teacher_student_error, q_state

    # offline training phase: train the student model using the qdagger loss
    distill_coeff = 1.0
    for global_step in track(range(args.offline_steps), description="offline student training"):
        data = rb.sample(args.batch_size)
        # perform a gradient-descent step
        loss, q_loss, q_pred, distill_loss, teacher_student_error, q_state = update(
            q_state,
            data.observations,
            data.actions,
            data.next_observations,
            data.rewards,
            data.terminations,
            distill_coeff,
        )

        # update the target network
        if global_step % args.target_network_frequency == 0:
            q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau))

        if global_step % 100 == 0:
            writer.add_scalar("offline/loss", jax.device_get(loss), global_step)
            writer.add_scalar("offline/td_loss", jax.device_get(q_loss), global_step)
            writer.add_scalar("offline/distill_loss", jax.device_get(distill_loss), global_step)
            writer.add_scalar("offline/q_values", jax.device_get(q_pred).sum(axis=-1).mean(), global_step)
            writer.add_scalar("offline/distill_coeff", distill_coeff, global_step)
            writer.add_scalar("offline/teacher_error", jax.device_get(teacher_student_error), global_step)

    # Continue using the old teacher replay buffer
    # rb = ReplayBuffer(
    #     args.buffer_size,
    #     envs.single_observation_space,
    #     envs.single_action_space,
    # )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    # )
    obs, _ = envs.reset(seed=args.seed)
    episodic_returns = deque(maxlen=10)
    autoreset = np.zeros((1,), dtype=np.bool_)

    # online training phase
    for global_step in track(range(args.total_timesteps), description="online student training"):
        # ALGO LOGIC: put action logic here
        # epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < args.end_e:  # epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network.apply(q_state.params, obs)
            actions = jax.device_get(q_values.argmax(axis=-1))

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer
        if not np.any(autoreset):
            rb.add(obs, actions, next_obs, rewards, terminated)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        autoreset = np.logical_or(terminated, truncated)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if np.any(autoreset):
            print(f"global_step={global_step}, episodic_return={infos['episode']['r'][0]}")
            writer.add_scalar("online/episodic_return", infos["episode"]["r"][0], global_step)
            writer.add_scalar("online/episodic_length", infos["episode"]["l"][0], global_step)
            # writer.add_scalar("charts/online/epsilon", epsilon, global_step)
            episodic_returns.append(infos["episode"]["r"][0])

        # ALGO LOGIC: training.
        # if global_step > args.learning_starts:   # remove as not removing teacher_rb
        if global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            # perform a gradient-descent step
            if len(episodic_returns) < 10:
                distill_coeff = 1.0
            else:
                distill_coeff = max(1 - np.mean(episodic_returns) / np.mean(teacher_episodic_returns), 0)
            loss, q_loss, q_pred, distill_loss, teacher_student_error, q_state = update(
                q_state,
                data.observations,
                data.actions,
                data.next_observations,
                data.rewards,
                data.terminations,
                distill_coeff,
            )

            if global_step % 100 == 0:
                writer.add_scalar("online/loss", jax.device_get(loss), global_step)
                writer.add_scalar("online/td_loss", jax.device_get(q_loss), global_step)
                writer.add_scalar("online/distill_loss", jax.device_get(distill_loss), global_step)
                writer.add_scalar("online/q_values", jax.device_get(q_pred).sum(axis=-1).mean(), global_step)
                writer.add_scalar("online/distill_coeff", distill_coeff, global_step)
                writer.add_scalar("online/teacher_error", jax.device_get(teacher_student_error), global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("online/SPS", int(global_step / (time.time() - start_time)), global_step)

        # update the target network
        if global_step % args.target_network_frequency == 0:
            q_state = q_state.replace(
                target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
            )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        # print(f"model saved to {model_path}")

        episodic_returns = evaluate(
            args.env_id, args.seed,
            video_folder=f'runs/{run_name}/videos/',
            epsilon=args.end_e,
            model=partial(ImpalaQNetwork, reward_vector_size=args.reward_vector_size),
            model_filename=model_path
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
