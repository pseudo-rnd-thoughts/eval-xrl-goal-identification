import json

import flax.serialization
import jax
import jax.numpy as jnp

from explanation_mechanisms.dqn_train import ImpalaQNetwork
from explanation_mechanisms.generate_explain_obs import load_explain_obs

if __name__ == "__main__":
    agent_ids = ("Dots-v2", "EnergyPill-v2", "Survival-v2", "LoseALife-v2")

    network_def = ImpalaQNetwork(action_dim=9)
    # env = make_env('MsPacman-Dots-v2', 0, 1, "", "")()
    # action_meanings = env.unwrapped.get_action_meanings()
    action_meanings = {
        0: "The Agent wants to stay still",
        1: "The Agent wants to move up",
        2: "The Agent wants to move right",
        3: "The Agent wants to move left",
        4: "The Agent wants to move down",
        5: "The Agent wants to move up and right",
        6: "The Agent wants to move up and left",
        7: "The Agent wants to move down and right",
        8: "The Agent wants to move down and left"
    }

    # explain observations for each agent
    explain_obs_data = load_explain_obs()
    explanations = {}
    for agent_id in agent_ids:
        with open(f'../models/MsPacman-{agent_id}_dqn_train.cleanrl_model', 'rb') as file:
            network_params = flax.serialization.from_bytes(
                network_def.init(jax.random.PRNGKey(1), jnp.zeros((1, 4, 84, 84), dtype=jnp.uint8)), file.read())

        for explain_obs in explain_obs_data:
            q_values = network_def.apply(network_params, jnp.expand_dims(explain_obs.agent_obs, axis=0))
            optimal_action = jax.device_get(jnp.argmax(q_values, axis=1))[0]
            explanations[f'obs-{explain_obs.id}-{agent_id}'] = action_meanings[optimal_action]

    with open("../explanations/optimal_action_explanation.json", "w") as file:
        json.dump(explanations, file)
