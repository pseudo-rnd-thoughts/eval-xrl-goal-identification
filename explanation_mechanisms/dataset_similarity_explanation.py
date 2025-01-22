import os

import flax
import flax.linen as nn
import jax
import matplotlib.animation
import numpy as np
import tensorflow as tf
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from explanation_mechanisms.dataset import load_observations, generate_dataset, load_renderings, combine_datasets, \
    split_into_episodes, load_episode_pos
from explanation_mechanisms.dqn_train import ImpalaQNetwork
from explanation_mechanisms.generate_explain_obs import load_explain_obs


class HiddenLayerAutoencoder(tf.keras.Model):

    def __init__(self, latent_dims: int):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(latent_dims),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(256),
        ])

    def call(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def learn_autoencoder(agent_id: str, training_epochs: int, training_batch_size: int, latent_dims: int,
                      test_size: float = 0.2, ):
    network_def = ImpalaQNetwork(action_dim=9)
    with open(f'../models/MsPacman-{agent_id}_dqn_train.cleanrl_model', "rb") as file:
        network_params = flax.serialization.from_bytes(
            network_def.init(jax.random.PRNGKey(1), np.zeros((1, 4, 84, 84))), file.read())

    if not os.path.exists(f"../datasets/dataset-similarity-explanation/combined-MsPacman/{agent_id}-hidden-activations.npz"):
        agent_observations = load_observations(f'../datasets/dataset-similarity-explanation/combined-MsPacman')
        print(f'`combined-MsPacman` shape={agent_observations.shape}')

        hidden_activations = np.zeros((len(agent_observations), 256))
        agent_observations = split_into_episodes(agent_observations, f'../datasets/dataset-similarity-explanation/combined-MsPacman')

        print(f'compute {agent_id} activations')
        i = 0
        for obs in tqdm(agent_observations):
            _, activations = network_def.apply(network_params, obs,
                                               capture_intermediates=lambda mdl, method_name: isinstance(mdl, nn.Dense))
            hidden_activations[i:i+len(obs)] = jax.device_get(activations["intermediates"]["Dense_0"]["__call__"][0])
            i += len(obs)

        del agent_observations

        num_hidden_activations = len(hidden_activations)
        hidden_activations = np.unique(hidden_activations, axis=0)
        num_unique_hidden_activations = len(hidden_activations)
        print(f'Removing duplicates - original={num_hidden_activations}, uniques={num_unique_hidden_activations}')

        np.savez(f"../datasets/dataset-similarity-explanation/combined-MsPacman/{agent_id}-hidden-activations.npz",
                 hidden_activations=hidden_activations,
                 num_hidden_activations=num_hidden_activations,
                 num_unique_hidden_activations=num_unique_hidden_activations)

    autoencoder = HiddenLayerAutoencoder(latent_dims=latent_dims)

    if not os.path.exists(f'../models/MsPacman-{agent_id}-autoencoder.weights.h5'):
        with np.load(f"../datasets/dataset-similarity-explanation/combined-MsPacman/{agent_id}-hidden-activations.npz") as file:
            hidden_activations = file["hidden_activations"]
        print(f'loading hidden-activations={hidden_activations.shape}')

        train_observations, test_observations = train_test_split(hidden_activations,
                                                                 test_size=test_size, random_state=123)
        print(f'{train_observations.shape=}, {test_observations.shape=}')

        autoencoder.compile(optimizer="adam", loss="mse")
        history = autoencoder.fit(train_observations, train_observations,
                                  epochs=training_epochs, shuffle=True, batch_size=training_batch_size,
                                  validation_data=(test_observations, test_observations))

        np.savez(f'../datasets/dataset-similarity-explanation/MsPacman-{agent_id}/autoencoder-training-results.npz',
                 loss=history.history['loss'], val_loss=history.history['val_loss'])
        autoencoder.save_weights(f'../models/MsPacman-{agent_id}-autoencoder.weights.h5')

    print(f'Computing {agent_id} embeddings')
    autoencoder.load_weights(f'../models/MsPacman-{agent_id}-autoencoder.weights.h5')
    dataset_observations = load_observations(f'../datasets/dataset-similarity-explanation/MsPacman-{agent_id}')
    hidden_activations = np.zeros((len(dataset_observations), 256))
    dataset_observations = split_into_episodes(dataset_observations,f'../datasets/dataset-similarity-explanation/MsPacman-{agent_id}')

    i = 0
    for obs in tqdm(dataset_observations):
        _, activations = network_def.apply(network_params, obs,
                                           capture_intermediates=lambda mdl, method_name: isinstance(mdl, nn.Dense))
        hidden_activations[i:i + len(obs)] = jax.device_get(activations["intermediates"]["Dense_0"]["__call__"][0])
        i += len(obs)

    embeddings = autoencoder.encode(hidden_activations)
    np.savez(f'../datasets/dataset-similarity-explanation/MsPacman-{agent_id}/embedding.npz', embeddings=embeddings)


def load_dataset_embeddings(folder_name: str):
    with np.load(f"{folder_name}/embedding.npz") as data:
        return data['embeddings']


def dataset_similarity_explanation(
    explanation_obs_embedding: np.ndarray,
    dataset_embeddings: np.ndarray,
    dataset_visualisations: np.ndarray,
    num_explanations: int = 2,
    explanation_length: int = 50,
    dataset_pos: np.ndarray | None = None
) -> np.ndarray:
    """Implementation of the dataset explanation with a provided explanation obs embedding
        and a dataset of embeddings, along with a dataset of observations to visualise the agent.

    Args:
        explanation_obs_embedding: The explanation observation embedding
        dataset_embeddings: The embedded dataset
        dataset_visualisations: The visualisation dataset observations
        num_explanations: The number of explanations generated
        explanation_length: The length of the explanations provided

    Returns:
        Numpy array for each explanation
    """
    assert isinstance(dataset_embeddings, np.ndarray), f'{type(dataset_embeddings)=}'
    assert isinstance(explanation_obs_embedding, np.ndarray), f'{type(explanation_obs_embedding)=}'
    assert dataset_embeddings.ndim == 2 and explanation_obs_embedding.ndim == 1, f'{dataset_embeddings.ndim=}, {explanation_obs_embedding.ndim=}'
    assert dataset_embeddings.shape[1] == explanation_obs_embedding.shape[0], f'{dataset_embeddings.shape=}, {explanation_obs_embedding.shape=}'

    if dataset_pos is None:
        dataset_pos = np.full(len(dataset_embeddings), explanation_length)

    # Compute the explanation obs embedding then distance between the embedded dataset and the embedded explanation obs
    explanation_dataset_distance = np.linalg.norm(
        dataset_embeddings - explanation_obs_embedding, axis=-1
    )
    assert explanation_dataset_distance.shape == (len(dataset_embeddings),)

    # We wish to generate num_explanations that do not contain the close by points, i.e., within explanation_length
    #   We find the minimum index then maximise the following explanation_length points
    #   To note, np.argpartition exists to efficiently find the minimum K indexes however
    #   as we wish to ignore the following indexes this doesn't work
    minimum_dataset_indexes = np.zeros(num_explanations, dtype=np.int32)
    max_distance = np.max(explanation_dataset_distance) + 1

    # Prevent sampling from observations at the end of an episode
    explanation_dataset_distance[dataset_pos < explanation_length] = max_distance

    # For each explanation find minimal explanation-dataset distance
    for num in range(num_explanations):
        minimum_dataset_indexes[num] = np.argmin(explanation_dataset_distance)
        explanation_dataset_distance[minimum_dataset_indexes[num] : minimum_dataset_indexes[num] + explanation_length] = max_distance

    # Using the minimum dataset indexes then we find the next explanation_length observations,
    #   this assumes that we have not reached the end of a trajectory which may not be true
    dataset_explanation_indexes = minimum_dataset_indexes[:, None] + np.arange(explanation_length)[None, :]
    assert dataset_explanation_indexes.shape == (num_explanations, explanation_length)

    # Get visualisations for the explanation indexes
    explanation_obs = dataset_visualisations[dataset_explanation_indexes]
    assert explanation_obs.shape == (num_explanations, explanation_length) + dataset_visualisations.shape[1:]

    return explanation_obs


if __name__ == "__main__":
    # save video name - dse-obs-{explain_obs}-agent-{agent_id}.mp4

    agent_ids = ("Dots-v2", "EnergyPill-v2", "Survival-v2", "LoseALife-v2")
    dataset_size = 10_000
    epsilon = 0.01

    explanation_length = 50
    num_explanations = 2
    pause = 15

    network_def = ImpalaQNetwork(action_dim=9)

    # generate dataset for each agent id
    for agent_id in agent_ids:
        if not os.path.exists(f'../datasets/dataset-similarity-explanation/MsPacman-{agent_id}'):

            with open(f'../models/MsPacman-{agent_id}_dqn_train.cleanrl_model', 'rb') as f:
                agent_params = flax.serialization.from_bytes(
                    network_def.init(jax.random.PRNGKey(0), np.zeros((1, 4, 84, 84))), f.read())

            generate_dataset(
                env_id=f'MsPacman-{agent_id}', network=network_def, params=agent_params,
                save_folder=f"../datasets/dataset-similarity-explanation/MsPacman-{agent_id}",
                dataset_size=dataset_size, num_vector_envs=16, epsilon=epsilon
            )

    # combine the datasets for learning the autoencoders
    if not os.path.exists('../datasets/dataset-similarity-explanation/combined-MsPacman'):
        combine_datasets(
            dataset_folders=[f"../datasets/dataset-similarity-explanation/MsPacman-{agent_id}"
                             for agent_id in agent_ids],
            new_folder='../datasets/dataset-similarity-explanation/combined-MsPacman'
        )

    # learn autoencoders
    for agent_id in agent_ids:
        if not os.path.exists(f'../datasets/dataset-similarity-explanation/MsPacman-{agent_id}/embedding.npz'):
            learn_autoencoder(agent_id, training_epochs=200, training_batch_size=128, latent_dims=16)

    # explain observations for each agent
    explain_obs_data = load_explain_obs()
    for agent_id in agent_ids:
        print(f'Generate explanations for {agent_id}')
        with open(f'../models/MsPacman-{agent_id}_dqn_train.cleanrl_model', 'rb') as f:
            network_params = flax.serialization.from_bytes(
                network_def.init(jax.random.PRNGKey(0), np.zeros((1, 4, 84, 84))), f.read())

        autoencoder = HiddenLayerAutoencoder(latent_dims=16)
        autoencoder.load_weights(f"../models/MsPacman-{agent_id}-autoencoder.weights.h5")

        dataset_embeddings = load_dataset_embeddings(f'../datasets/dataset-similarity-explanation/MsPacman-{agent_id}')
        dataset_renderings = load_renderings(f'../datasets/dataset-similarity-explanation/MsPacman-{agent_id}')
        dataset_renderings = dataset_renderings[:, 1:172]
        dataset_pos = load_episode_pos(f'../datasets/dataset-similarity-explanation/MsPacman-{agent_id}', reverse=True)

        for explain_obs in tqdm(explain_obs_data):
            _, activations = network_def.apply(network_params, np.expand_dims(explain_obs.agent_obs, axis=0),
                                               capture_intermediates=lambda mdl, _: isinstance(mdl, nn.Dense))
            hidden_activations = jax.device_get(activations["intermediates"]["Dense_0"]["__call__"][0])
            explain_obs_embedding = np.array(autoencoder.encode(hidden_activations))[0]

            explanation_videos = dataset_similarity_explanation(
                explain_obs_embedding, dataset_embeddings, dataset_renderings,
                num_explanations=num_explanations, explanation_length=explanation_length, dataset_pos=dataset_pos
            )
            print(f'{explanation_videos.shape=}')

            fig, ax = plt.subplots(ncols=1, figsize=(3.5, 4))
            ax.axis('off')
            explanation_title = ax.text(45, -10, "Example 1", fontsize="xx-large")
            explanation_imshow = ax.imshow(explanation_videos[0, 0])
            plt.tight_layout()
            explanation_num = 0

            def update(t):
                global explanation_num
                if t == explanation_length + pause:
                    explanation_num = 1
                    explanation_title.set_text("Example 2")
                    explanation_imshow.set_data(explanation_videos[1, 0])
                    return explanation_title, explanation_imshow

                if t > explanation_length + pause:
                    t = t - (explanation_length + pause)
                if t < pause:
                    return
                explanation_imshow.set_data(explanation_videos[explanation_num, t - pause])
                return explanation_imshow,

            anim_video = matplotlib.animation.FuncAnimation(
                fig, update,
                frames=np.arange((explanation_length+pause) * 2), interval=1000//10, blit=False
            )
            anim_video.save(f"../explanations/dataset-similarity-explanation/dse-obs-{explain_obs.id}-agent-{agent_id}.gif")
            plt.close(fig)

            # video = ImageSequenceClip(list(explanation_videos.reshape((100, 171, 160, 3))), fps=15)
            # video.write_gif(f"../explanations/dataset-similarity-explanation/dse-obs-{explain_obs.id}-agent-{agent_id}.gif", logger=None)
