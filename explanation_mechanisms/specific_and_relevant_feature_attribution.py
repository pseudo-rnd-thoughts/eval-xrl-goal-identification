import cv2
import flax
import jax
import jax.numpy as jnp
import numpy as onp
from PIL import Image
from flax import linen as nn
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
from tqdm import tqdm

from explanation_mechanisms.dqn_train import ImpalaQNetwork
from explanation_mechanisms.generate_explain_obs import load_explain_obs


def _generate_perturbation_mask(
    center: tuple[int, int], mask_points: onp.ndarray, radius: int = 25
):
    """Generates the saliency mask

    Args:
        center: The mask center position
        mask_points: Numpy array of mask points
        radius: The radius of the mask

    Returns:
        Normalised mask over the mask points at the center (mean) and radius (standard deviation)
    """
    mask = multivariate_normal.pdf(mask_points, center, radius)
    return mask / mask.max()


def gen_specific_and_relevant_feature_attribution(
    obs: onp.ndarray,
    network_def: nn.Module,
    network_params: flax.core.FrozenDict,
    perturbation_spacing: int = 5,
) -> tuple[onp.ndarray, onp.ndarray]:
    """
    Generate the saliency map for a particular observation and JAX network definition following Puri et al., 2020

    Args:
        obs: observations of the agent
        network_def: Network definition
        network_params: Network parameters
        perturbation_spacing: The perturbation patch size, this speeds up the algorithm. For optimal saliency map
        use a spacing of 1, the default is 5 middle ground

    Returns:
         Tuple of the saliency map and the raw saliency values in a 2d array
    """
    if obs.ndim == 3:
        obs = onp.expand_dims(obs, 0)

    assert obs.shape == (1, 4, 84, 84), f'{obs.shape=}'
    assert 0 <= perturbation_spacing, "perturbation spacing must be greater than zero"

    # Finds the blurred observations and mask points for generating the mask
    blurred_obs = gaussian_filter(obs, sigma=3)
    mask_points = onp.dstack(onp.meshgrid(onp.arange(84), onp.arange(84)))
    assert blurred_obs.shape == (1, 4, 84, 84) and mask_points.shape == (84, 84, 2), (
        f"Blurred obs: {blurred_obs.shape} - expected (1, 4, 84, 84) and "
        f"mask points: {mask_points.shape} - expected (84, 84, 2)"
    )

    perturbation_per_row = 84 // perturbation_spacing + 1

    # For each point on the observation, generating the perturbation
    #   This allows for all the perturbed observation to be run in parallel with vmap network_q_values
    perturbed_obs = onp.zeros((perturbation_per_row**2, 4, 84, 84))
    for x, x_center in enumerate(onp.arange(0, 84, perturbation_spacing)):
        for y, y_center in enumerate(onp.arange(0, 84, perturbation_spacing)):
            # calculate the mask for the center points and repeat for 4 dimensions to match the actual obs (84, 84, 4)
            mask = _generate_perturbation_mask(
                (x_center, y_center), mask_points, perturbation_spacing**2
            )
            obs_mask = onp.repeat(mask[onp.newaxis, :, :], 4, axis=0)

            # Compute the perturbed observation using the equation from the paper
            perturbed_obs[y * perturbation_per_row + x] = (
                obs * (1 - obs_mask) + blurred_obs * obs_mask
            )

    # assert onp.all(perturbed_ob != obs for perturbed_ob in perturbed_obs)

    # calculate the true q-values and perturbed q-values
    true_q_values = network_def.apply(network_params, obs)
    perturbed_q_values = network_def.apply(network_params, perturbed_obs)

    # specify change
    action = jnp.argmax(true_q_values[0])
    true_p = jnp.exp(true_q_values[0, action]) / jnp.sum(jnp.exp(true_q_values))
    perturbed_p = jnp.exp(perturbed_q_values[:, action]) / jnp.sum(jnp.exp(perturbed_q_values), axis=1)
    delta_p = true_p - perturbed_p

    # relevant changes
    true_p_rem = true_q_values[:, jnp.where(jnp.arange(true_q_values.shape[1]) != action)[0]]
    perturbed_p_rem = perturbed_q_values[:, jnp.where(jnp.arange(true_q_values.shape[1]) != action)[0]]
    kl_diverg = jnp.sum(true_p_rem * jnp.log(jnp.clip(true_p_rem / perturbed_p_rem, min=1e-6)), axis=1)
    similarity = 1 / (1 + kl_diverg)  # K

    saliency = jax.device_get((2 * similarity * delta_p) / (similarity + delta_p))
    assert saliency.shape == (perturbation_per_row ** 2,)
    assert not onp.isnan(saliency).any()

    # calculate the 2d saliency map based on the saliency list
    saliency_map = onp.zeros((84, 84), dtype=onp.float32)
    saliency_values = onp.zeros((84 // perturbation_spacing + 1, 84 // perturbation_spacing + 1), dtype=onp.float32)
    for x, x_center in enumerate(onp.arange(0, 84, perturbation_spacing)):
        for y, y_center in enumerate(onp.arange(0, 84, perturbation_spacing)):
            mask = _generate_perturbation_mask(
                (x_center, y_center), mask_points, perturbation_spacing**2
            )
            saliency_map += saliency[y * perturbation_per_row + x] * mask
            saliency_values[
                y_center // perturbation_spacing, x_center // perturbation_spacing
            ] = saliency[y * perturbation_per_row + x]

    # for the map, normalise the saliency values to between 0 and 255
    #   while the values are the raw saliency values (84, 84). For a list of saliency values, .flatten()
    normalised_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min()) * 255
    return (
        normalised_saliency_map.astype(onp.uint8),
        saliency_values,
    )


if __name__ == "__main__":
    # save image name = pbsm-obs-{explain_obs}-agent-{agent_id}.png
    explain_obs_data = load_explain_obs()

    network_def = ImpalaQNetwork(action_dim=9)
    for agent_id in ("Dots-v2", "EnergyPill-v2", "Survival-v2", "LoseALife-v2"):
        print(f'Generating explanations for {agent_id}')
        with open(f"../models/MsPacman-{agent_id}_dqn_train.cleanrl_model", "rb") as file:
            network_params = flax.serialization.from_bytes(
                network_def.init(jax.random.PRNGKey(0), onp.zeros((1, 4, 84, 84))), file.read())

        for explain_obs in tqdm(explain_obs_data):
            saliency_map, _ = gen_specific_and_relevant_feature_attribution(explain_obs.agent_obs, network_def, network_params)

            # incorporate the saliency map with the environment render
            resized_saliency_map = cv2.resize(saliency_map, (160, 210))
            heatmap = cv2.applyColorMap(resized_saliency_map.astype(onp.uint8), cv2.COLORMAP_JET)

            render_saliency = cv2.addWeighted(explain_obs.env_render, 0.7, heatmap, 0.3, 0)
            render_saliency = cv2.cvtColor(render_saliency, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(render_saliency[1:172])
            image.save(f"../explanations/specific_and_relevant_feature_attribution/sarfa-obs-{explain_obs.id}-agent-{agent_id}.png")
