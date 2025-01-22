import os
from typing import NamedTuple

import flax
import flax.linen as nn
import jax
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from explanation_mechanisms.dataset import generate_dataset, load_observations, load_renderings
from explanation_mechanisms.dqn_train import ImpalaQNetwork


class ExplainObsData(NamedTuple):
    id: int                 # easy way to refer to the observation
    agent_id: str           # how the observation was generated, which agent and env
    dataset_pos: int        # the dataset position
    metadata: dict          # the normalised episode position
    agent_obs: np.ndarray   # the agent observation (1, 84, 84, 4)
    env_render: np.ndarray  # the environment (210, 180, 3)


def generate_explanations(num_explanations: int,
                          selector: str,
                          quantile: float = 0.75):
    print(f'Generating explanations with {selector=}')

    # Config data
    agent_ids = ("Dots-v2", "EnergyPill-v2", "Survival-v2", "LoseALife-v2")
    episode_nums = (5, 5, 5, 8)

    # Load all the agents
    network_def = ImpalaQNetwork(action_dim=9)
    network_params = {}
    for agent_id in agent_ids:
        with open(f'../models/MsPacman-{agent_id}_dqn_train.cleanrl_model', "rb") as file:
            params = network_def.init(jax.random.PRNGKey(0), np.zeros((1, 4, 84, 84)))
            network_params[agent_id] = flax.serialization.from_bytes(params, file.read())

    from explanation_mechanisms.dataset_similarity_explanation import HiddenLayerAutoencoder

    agent_autoencoder = {}
    for agent_id in agent_ids:
        model = HiddenLayerAutoencoder(latent_dims=16)
        model.load_weights(f'../models/MsPacman-{agent_id}-autoencoder.weights.h5')
        agent_autoencoder[agent_id] = model

    if not os.path.exists('../../eval-xrl-goal-identification/explanations /explanation-obs'):
        os.mkdir('../../eval-xrl-goal-identification/explanations /explanation-obs')

    # For each goal, generate explanations
    explanations = []
    explanation_embeddings = {sub_agent_id: [] for sub_agent_id in agent_ids}
    for agent_id, num_episodes in zip(agent_ids, episode_nums):
        # generate explanation dataset if missing
        if not os.path.exists(f"../datasets/explanations/MsPacman-{agent_id}"):
            print(f'Generating dataset for {agent_id} with {num_episodes} episodes')
            generate_dataset(
                f"MsPacman-{agent_id}",
                network_def, network_params[agent_id],
                save_folder=f"../datasets/explanations/MsPacman-{agent_id}",
                dataset_episodes=num_episodes, num_vector_envs=1, seed=3141
            )

        episode_observations = load_observations(f"../datasets/explanations/MsPacman-{agent_id}")
        episode_renders = load_renderings(f"../datasets/explanations/MsPacman-{agent_id}")
        episode_len = len(episode_observations)
        # print(f'MsPacman {agent_id} explanation dataset size: {episode_len}')
        assert episode_len > 0, f'{type(episode_observations)=}'

        # compute the importance and embeddings for each agent
        if not os.path.exists(f'../datasets/explanations/MsPacman-{agent_id}/q-values-activations.npz'):
            obs_importance, agent_obs_embeddings = {}, {}
            for sub_agent_id in agent_ids:
                q_values, activations = network_def.apply(network_params[sub_agent_id],  episode_observations,
                                                          capture_intermediates=lambda mdl, _: isinstance(mdl, nn.Dense))

                importance = q_values.max(axis=-1) - q_values.min(axis=-1)
                obs_importance[sub_agent_id] = jax.device_get(importance)

                hidden_activations = jax.device_get(activations["intermediates"]["Dense_0"]["__call__"][0])
                agent_obs_embeddings[sub_agent_id] = np.array(agent_autoencoder[sub_agent_id].encode(hidden_activations))

            np.savez(f'../datasets/explanations/MsPacman-{agent_id}/q-values-activations.npz',
                     obs_importance=obs_importance, agent_obs_embeddings=agent_obs_embeddings)

        with np.load(f'../datasets/explanations/MsPacman-{agent_id}/q-values-activations.npz', allow_pickle=True) as file:
            obs_importance, agent_obs_embeddings = file["obs_importance"].item(), file["agent_obs_embeddings"].item()
        print(f'Selecting explanations for {agent_id}')

        obs_importance = np.array([obs_importance[sub_agent_id] / np.max(obs_importance[sub_agent_id]) for sub_agent_id in agent_ids])
        avg_obs_importance = np.mean(obs_importance, axis=0)
        # print(f'obs_importance={obs_importance.shape}, avg_obs_importance={avg_obs_importance.shape}, '
        #       f'embedding={agent_obs_embeddings[agent_id].shape}')

        # Plot the importance
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.suptitle(f"{agent_id} importance")
        for i, sub_agent_id in enumerate(agent_ids):
            ax.plot(np.arange(episode_len), obs_importance[i], label=sub_agent_id)
        ax.plot(np.arange(episode_len), avg_obs_importance, label="average")
        ax.legend()
        plt.savefig(f'../datasets/explanations/MsPacman-{agent_id}/agent-importance.png')
        plt.close(fig)
        # plt.show()

        # generate the explanations
        for i in range(num_explanations):
            # compute observation distance to explanations
            if len(explanations) == 0:
                avg_obs_distance = np.random.uniform(0, 1, size=len(episode_observations))  # randomly select the first observation
            else:
                embedding_distance = np.array([  # [num-agents, num-explanations, num-observations]
                    [
                        np.linalg.norm(agent_obs_embeddings[sub_agent_id] - explanation_embeddings[sub_agent_id][i], axis=1)
                        for i in range(len(explanations))
                    ]
                    for sub_agent_id in agent_ids
                ])
                assert embedding_distance.shape == (len(agent_ids), len(explanations), len(episode_observations))

                min_explanations_distance = embedding_distance.transpose((2, 0, 1)).min(axis=2).transpose((1, 0))  # [agents, num-observations]
                obs_distance = min_explanations_distance / min_explanations_distance.max(axis=1, keepdims=True)  # [agents, num-observations]
                avg_obs_distance = np.mean(obs_distance, axis=0)

            # select the argmax of masked average importance
            if selector == "importance":
                mask = np.ones(len(episode_observations), dtype=bool)
                for e in explanations:
                    if agent_id in e.agent_id:
                        mask[e.dataset_pos] = False
                pos = np.argmax(avg_obs_importance * mask)
            elif selector == "importance-mask":
                mask = np.ones(len(episode_observations), dtype=bool)
                for e in explanations:
                    if agent_id in e.agent_id:
                        mask[np.clip(np.arange(e.dataset_pos - 10, e.dataset_pos + 10),
                                     0, len(episode_observations))] = False
                pos = np.argmax(avg_obs_importance * mask)
            elif selector == "embedding-distance":
                pos = np.argmax(avg_obs_distance)
            elif selector == "embedding-similarity":
                if len(explanations) == 0:
                    avg_obs_similarity = np.random.uniform(0, 1, size=len(episode_observations))
                else:
                    cosine_similarity = 1 - np.array([
                        [
                            (agent_obs_embeddings[sub_agent_id] @ explanation_embeddings[sub_agent_id][i]) / (
                                    np.linalg.norm(agent_obs_embeddings[sub_agent_id], axis=1) * np.linalg.norm(explanation_embeddings[sub_agent_id][i]))
                            for i in range(len(explanations))
                        ]
                        for sub_agent_id in agent_ids
                    ])
                    min_similarity = cosine_similarity.transpose((2, 0, 1)).min(axis=2).transpose((1, 0))
                    obs_similarity = min_similarity / min_similarity.max(axis=1, keepdims=True)
                    avg_obs_similarity = np.mean(obs_similarity, axis=0)

                pos = np.argmax(avg_obs_similarity)
            elif selector == "importance-top-embedding":
                pos = np.argmax(avg_obs_importance * (np.percentile(avg_obs_distance, quantile) < avg_obs_distance))
            elif selector == "importance-embedding-distance":
                # pos = np.argmax(np.mean(obs_importance * obs_distance, axis=0))  # compute the mean over the agents
                pos = np.argmax(avg_obs_importance * avg_obs_distance)
            else:
                raise ValueError(f'Unknown {selector=}')

            print(f'\tnum={i:02}, pos={pos:04}, '
                  f'importance={float(avg_obs_importance[pos]):.3f}, '
                  f'distance={float(avg_obs_distance[pos]):.3f}')

            # Add explanations
            explanations.append(
                ExplainObsData(
                    id=len(explanations),
                    agent_id=f"MsPacman-{agent_id}",
                    dataset_pos=pos,
                    agent_obs=episode_observations[pos],
                    env_render=episode_renders[pos],
                    metadata={
                        'importance': avg_obs_importance[pos],
                        'distance': avg_obs_distance[pos],
                    }
                )
            )
            # Save obs as image
            Image.fromarray(episode_renders[pos][1:172]).save(f'../explanations/explanation-obs/obs-{len(explanations)-1}.png')

            # Add explanation to embedding list
            for sub_agent_id in agent_ids:
                explanation_embeddings[sub_agent_id].append(agent_obs_embeddings[sub_agent_id][pos])

        fig, axs = plt.subplots(nrows=5, ncols=num_explanations // 5, figsize=(8, 15))
        fig.suptitle(agent_id)
        for ax, explanation in zip(axs.flatten(), explanations[-num_explanations:], strict=True):
            ax.imshow(explanation.env_render[1:172])
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'../datasets/explanations/MsPacman-{agent_id}/explanation-renders.png')
        plt.close(fig)

    print('Saving all explanations')
    explanation_embeddings = {sub_agent_id: np.array(embeddings) for sub_agent_id, embeddings in explanation_embeddings.items()}
    avg_explanation_importance = np.mean([explanation.metadata["importance"] for explanation in explanations])
    greed_avg_explanation_distance = np.mean([explanation.metadata["distance"] for explanation in explanations])

    final_embedding_distance = np.array([  # [num-agents, num-explanations, num-explanations - 1]
        [
            np.linalg.norm(
                explanation_embeddings[sub_agent_id][i] -
                explanation_embeddings[sub_agent_id][np.where(np.arange(len(explanations)) != i)[0]],
                axis=1
            )
            for i in range(len(explanations))
        ]
        for sub_agent_id in agent_ids
    ])
    assert final_embedding_distance.shape == (len(agent_ids), len(explanations), len(explanations) - 1)
    final_min_explanations_distance = final_embedding_distance.transpose((2, 0, 1)).min(axis=2).transpose((1, 0))  # [num-agents, num-explanations]
    final_obs_distance = final_min_explanations_distance / final_min_explanations_distance.max(axis=1, keepdims=True)  # [num-agents, num-explanations]
    final_avg_explanation_distance = np.mean(final_obs_distance, axis=0).mean()  # []

    print(f'Average explanation importance={avg_explanation_importance:.3f}')
    print(f'Average greedy explanation distance={greed_avg_explanation_distance:.3f}')
    print(f'Average final explanation distance={final_avg_explanation_distance:.3f}')
    np.savez(
        f'../../eval-xrl-goal-identification/explanations /explain-obs.npz' ,
        **dict(zip(ExplainObsData._fields, zip(*explanations))),
        selector=selector,
        avg_explanation_importance=avg_explanation_importance,
        greedy_avg_explanation_distance=greed_avg_explanation_distance,
        final_avg_explanation_distance=final_avg_explanation_distance,
    )


def load_explain_obs() -> list[ExplainObsData]:
    with np.load("../../eval-xrl-goal-identification/explanations /explain-obs.npz", allow_pickle=True) as file:
        explanations = [ExplainObsData(*data) for data in zip(*[file[prop] for prop in ExplainObsData._fields])]
        return explanations


if __name__ == "__main__":
    # generate_explanations(num_explanations=10, selector="importance")
    # generate_explanations(num_explanations=10, selector="importance-mask")
    # generate_explanations(num_explanations=10, selector="embedding-distance")
    # generate_explanations(num_explanations=10, selector="embedding-similarity")
    # generate_explanations(num_explanations=10, selector="importance-top-embedding")
    generate_explanations(num_explanations=5, selector="importance-embedding-distance")

# number of explanations = 5
# | Selector                      | Avg importance | Greedy Obs Distance | Final Obs Distance |
# |-------------------------------|----------------|---------------------|--------------------|
# | Importance                    | 0.476          | 0.244               | 0.162              |
# | Importance-mask               | 0.452          | 0.303               | 0.213              |
# | Embedding-distance            | 0.234          | 0.765               | 0.372              |
# | Embedding similarity          | 0.258          | 0.598               | 0.336              |
# | Importance top embedding      | 0.474          | 0.259               | 0.172              |
# | Importance embedding distance | 0.408          | 0.527               | 0.296              |
