import functools
import json
import os
import shutil

import gymnasium as gym
import numpy as onp
from tqdm import tqdm


def generate_dataset(
    env_id: str,
    network,
    params,
    save_folder: str,
    dataset_size: int = None,
    dataset_episodes: int = None,
    num_vector_envs: int = 16,
    epsilon: float = 0.05,
    seed: int | None = None,
) -> tuple[int, int, int]:
    """Generates a training dataset using an agent for an environment, actions are selected randomly
        (but not saved to the dataset) epsilon percent of the time

    Args:
        env_id: Environment id
        network: The network model
        params: Network parameters
        save_folder: The folder where the trajectory is saved to
        dataset_size: The size of the dataset generate
        dataset_episodes: The number of dataset episodes to generate
        num_vector_envs: The number of environments to run in parallel
        epsilon: The probability of a random action being taken, default is 1%
        seed: The seed for the vector environments

    Returns:
        The number of episodes run, the total number of steps taken and the total number of steps saved
    """
    os.makedirs(save_folder, exist_ok=True)

    import jax
    import jax.numpy as jnp

    assert (dataset_size is not None) != (dataset_episodes is not None), \
        f"Expects that either dataset size ({dataset_size}) or dataset episodes ({dataset_size}) are not None"

    # Initial the environments and the initial observations
    envs = gym.make_vec(env_id, num_envs=num_vector_envs, render_mode="rgb_array")
    obs, info = envs.reset(seed=seed)
    envs.action_space.seed(seed=seed)
    rendering = envs.render()

    # obs, rendering, actions, and reward for each environment to track transitions
    env_datasets = [([], [], [], []) for _ in range(num_vector_envs)]

    # Generate a progress bar as the process can take a while depending on the dataset_size
    progress_bar = tqdm(mininterval=5, desc=f"{env_id=}")
    # Save the total steps taken, the number of steps saved so far and
    #   the total number of trajectories / episodes finished
    steps_taken, saved_steps, episode_num = 0, 0, 0
    autoreset = onp.zeros(num_vector_envs, dtype=onp.bool_)
    while (dataset_size is not None and saved_steps < dataset_size) or (
        dataset_episodes is not None and episode_num < dataset_episodes
    ):
        # Select the actions for the observations
        q_values = network.apply(params, obs)
        optimal_actions = jax.device_get(jnp.argmax(q_values, axis=1))
        random_actions = envs.action_space.sample()
        actions = onp.where(onp.random.random(num_vector_envs) > epsilon,
                            optimal_actions, random_actions)

        assert q_values.shape == (num_vector_envs, envs.single_action_space.n)
        assert actions.shape == (envs.num_envs,), actions.shape

        next_obs, reward, terminated, truncated, info = envs.step(actions)

        for env_num in range(num_vector_envs):
            steps_taken += 1

            if not autoreset[env_num]:
                # todo - this doesn't correctly save the last obs / render from an episode, use minari
                env_datasets[env_num][0].append(obs[env_num])
                env_datasets[env_num][1].append(rendering[env_num])
                env_datasets[env_num][2].append(actions[env_num])
                env_datasets[env_num][3].append(reward[env_num])

            # If the environment terminates then save the trajectory
            if terminated[env_num] or truncated[env_num]:
                with open(f"{save_folder}/episode-{episode_num}.npz", "wb") as file:
                    onp.savez_compressed(
                        file,
                        observations=onp.array(env_datasets[env_num][0]),
                        renderings=onp.array(env_datasets[env_num][1]),
                        actions=onp.array(env_datasets[env_num][2]),
                        rewards=onp.array(env_datasets[env_num][3]),
                        length=len(env_datasets[env_num][0]),
                        terminated=terminated[env_num],
                        truncated=truncated[env_num],
                    )

                # update the saved_steps and episodes_runs then reset the environment dataset
                saved_steps += len(env_datasets[env_num][0])
                episode_num += 1
                env_datasets[env_num] = ([], [], [], [])

        # Update the observations with the new observation
        obs = next_obs
        autoreset = onp.logical_or(terminated, truncated)
        rendering = envs.render()

        progress_bar.update(num_vector_envs)
    # Close the progress bar
    progress_bar.close()

    # When the dataset is generated then save the metadata
    with open(f"{save_folder}/metadata.json", "w") as file:
        json.dump(
            {
                "env-id": env_id,
                "dataset-size": dataset_size,
                "dataset-episodes": dataset_episodes,
                "num-envs": num_vector_envs,
                "episodes-run": episode_num,
                "steps-taken": steps_taken,
                "saved-steps": saved_steps,
                "seed": seed,
            },
            file,
        )

    # Return the metadata
    return episode_num, steps_taken, saved_steps


def _load_dataset_prop(
    dataset_folder: str,
    prop: str,
    dtype: str = None,
    num_files: int | None = None,
    size: int | None = None,
) -> onp.ndarray:
    """Loads a dataset from dataset folder with property, dtype, num_files to load and dataset size

    :param dataset_folder: The folder to load the dataset from
    :param prop: The property to load
    :param dtype: The dtype from the dataset
    :param num_files: The number of files to load
    :param size: The size of the dataset
    :return: The loaded dataset
    """
    assert not (
        (num_files is not None) and (size is not None)
    ), f"Both {num_files} and {size} are not None, only allow one of the parameters to be not None"
    assert (
        num_files is None or 0 < num_files
    ), f"Number of files ({num_files}) must be None or greater than zero"
    assert size is None or 0 < size, f"Size ({size}) must be None or greater than zero"

    filenames = sorted(
        (
            filename
            for filename in os.listdir(dataset_folder)
            if "episode-" in filename and ".npz" in filename
        ),
        key=lambda filename: int(
            filename.replace("episode-", "").replace(".npz", "")
        ),
    )
    if len(filenames) == 0:
        return onp.array([])

    if size is not None:
        with onp.load(f'{dataset_folder}/{filenames[0]}', allow_pickle=True) as file:
            data_shape = file[prop].shape

        dataset = onp.empty(shape=(size,) + data_shape[1:], dtype=dtype)

        dataset_size, pos = 0, 0
        while dataset_size < size:
            with onp.load(f"{dataset_folder}/{filenames[pos]}", allow_pickle=True) as file:
                if dataset_size + file["length"] <= size:
                    dataset[dataset_size: dataset_size + file["length"]] = file[prop]
                    dataset_size += file["length"]
                    pos += 1
                else:
                    dataset[dataset_size:] = file[prop][:size - dataset_size]
                    dataset_size = size
    else:
        if num_files is not None:
            filenames = filenames[:num_files]

        dataset_size, data_shape = 0, (-1,)
        for filename in filenames:
            with onp.load(f"{dataset_folder}/{filename}", allow_pickle=True) as file:
                dataset_size += file["length"]

                if data_shape == (-1,):
                    data_shape = file[prop].shape

        dataset = onp.empty(shape=(dataset_size,) + data_shape[1:], dtype=dtype)

        pos = 0
        for filename in filenames:
            with onp.load(f"{dataset_folder}/{filename}") as file:
                length = file["length"]
                dataset[pos : pos + length] = file[prop]
                pos += length

    return dataset


load_observations = functools.partial(_load_dataset_prop, prop="observations", dtype=onp.uint8)
load_renderings = functools.partial(_load_dataset_prop, prop="renderings", dtype=onp.uint8)
load_actions = functools.partial(_load_dataset_prop, prop="actions", dtype=onp.int32)
load_rewards = functools.partial(_load_dataset_prop, prop="rewards")


def split_into_episodes(data, dataset_folder: str,):
    filenames = sorted(
        (
            filename
            for filename in os.listdir(dataset_folder)
            if "episode-" in filename and ".npz" in filename
        ),
        key=lambda filename: int(
            filename.replace("episode-", "").replace(".npz", "")
        ),
    )
    print(f'{filenames}')

    episode_data = []
    episode_start = 0
    for filename in filenames:
        with onp.load(f"{dataset_folder}/{filename}", allow_pickle=True) as file:
            episode_length = file["length"]

            episode_data.append(data[episode_start: episode_start + episode_length])
            episode_start += episode_length

    return episode_data


def load_episode_pos(dataset_folder: str, normalised: bool = False, reverse: bool = False):
    """Loads the episode position for each of the data points in the dataset.
    This can be normalised between [0, 1] to easier understanding."""
    filenames = sorted(
        (
            filename
            for filename in os.listdir(dataset_folder)
            if "episode-" in filename and ".npz" in filename
        ),
        key=lambda filename: int(
            filename.replace("episode-", "").replace(".npz", "")
        ),
    )

    dataset_size = 0
    for filename in filenames:
        with onp.load(f"{dataset_folder}/{filename}") as file:
            dataset_size += file["length"]

    dataset = onp.empty(shape=(dataset_size,), dtype=onp.float32 if normalised else onp.int32)
    pos = 0
    for filename in filenames:
        with onp.load(f"{dataset_folder}/{filename}") as file:
            length = file["length"]
            if normalised:
                if reverse:
                    dataset[pos: pos + length] = onp.linspace(1, 0, length)
                else:
                    dataset[pos: pos + length] = onp.linspace(0, 1, length)
            else:
                if reverse:
                    dataset[pos: pos + length] = onp.arange(length, 0, -1)
                else:
                    dataset[pos: pos + length] = onp.arange(length)

            pos += length

    return dataset


def combine_datasets(dataset_folders, new_folder):
    os.mkdir(new_folder)

    combined_metadata = {
        "env-id": [],
        "dataset-size": [],
        "dataset-episodes": [],
        "num-envs": [],
        "episodes-run": 0,
        "steps-taken": 0,
        "saved-steps": 0,
        "seed": [],
        "combined-datasets": dataset_folders
    }
    episode_num = 0
    for folder in dataset_folders:
        filenames = sorted(
            (
                filename
                for filename in os.listdir(folder)
                if "episode-" in filename and ".npz" in filename
            ),
            key=lambda filename: int(
                filename.replace("episode-", "").replace(".npz", "")
            ),
        )

        for filename in filenames:
            shutil.copy(f'{folder}/{filename}', f'{new_folder}/episode-{episode_num}.npz')
            episode_num += 1

        # update metadata
        with open(f'{folder}/metadata.json') as file:
            folder_metadata = json.load(file)

            combined_metadata["env-id"].append(folder_metadata["env-id"])
            combined_metadata["dataset-size"].append(folder_metadata["dataset-size"])
            combined_metadata["dataset-episodes"].append(folder_metadata["dataset-episodes"])
            combined_metadata["num-envs"].append(folder_metadata["num-envs"])
            combined_metadata["episodes-run"] += folder_metadata["episodes-run"]
            combined_metadata["steps-taken"] += folder_metadata["steps-taken"]
            combined_metadata["saved-steps"] += folder_metadata["saved-steps"]
            combined_metadata["seed"].append(folder_metadata["seed"])

    # write the metadata
    with open(f'{new_folder}/metadata.json', 'w') as file:
        json.dump(combined_metadata, file)
