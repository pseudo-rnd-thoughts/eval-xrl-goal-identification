import json
from datetime import datetime
from textwrap import TextWrapper

import flax
import jax
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from explanation_mechanisms.generate_explain_obs import load_explain_obs
from explanation_mechanisms.trd_dqn_train import ImpalaQNetwork as TrdQNetwork

no_domain_knowledge_prompt = [
    {
        "role": "system",
        "content": [{
            "text": "You are an assistant providing summarises of an agent's, playing a sequential game, "
                    "predicting their future rewards for the next few timesteps. "
                    "Help describe in a one-sentence summary the pattern of rewards, highlighting how far in the future this happens, "
                    "if the pattern is periodic or consistent, positive or negative, etc. "
                    "Ignore rewards close to zero compared to the rest of the pattern, as this is related to data noise.",
            "type": "text"
        }]
    },
    {"role": "user", "content": [{"type": "text", "text": "[0, 0, 0, 0, 4, 0, 0]"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "The agent expects a reward of 4 in 5 timesteps"}]},
    {"role": "user", "content": [{"type": "text", "text": "[1, 1, 1, 1, 1, 1]"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "The agent expects consistent positive rewards of 1"}]},
]


if __name__ == "__main__":
    # trd-obs-{explain_obs}-agent-{agent_id}.png
    explain_obs_data = load_explain_obs()

    with open('../openai.txt') as file:
        client = OpenAI(organization=None, project=None, api_key=file.read())

    network_def = TrdQNetwork(action_dim=9, reward_vector_size=41)
    discount_factor = np.power(0.99, np.arange(40))
    text_wrapper = TextWrapper(width=60)

    explanations = {}
    for agent_id in ("Dots-v2", "EnergyPill-v2", "Survival-v2", "LoseALife-v2"):
        print(f'Generating explanations for {agent_id}')
        explanations[agent_id] = []

        with open(f'../models/MsPacman-{agent_id}_trd_dqn_train.cleanrl_model', 'rb') as file:
            params = flax.serialization.from_bytes(
                network_def.init(jax.random.PRNGKey(0), np.zeros((1, 4, 84, 84))), file.read())

        for explain_obs in tqdm(explain_obs_data):
            trd_q_values = network_def.apply(params, np.expand_dims(explain_obs.agent_obs, axis=0),
                                             method=TrdQNetwork.decomposed_q_value)

            optimal_action = np.array(trd_q_values.sum(axis=-1).argmax(axis=-1))[0]
            future_expected_rewards = np.array(trd_q_values[0, optimal_action, :-1]) / discount_factor

            obs_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{np.round(future_expected_rewards, 2)}"
                    }
                ]
            }

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=no_domain_knowledge_prompt + [obs_message],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={"type": "text"}
            )
            reward_summary = response.choices[0].message.content
            print(f'{reward_summary}')
            explanations[agent_id].append(reward_summary)

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(np.arange(40)+1, np.round(future_expected_rewards, 2))
            ax.set_title("\n".join(text_wrapper.wrap(reward_summary)))
            ax.set_xlabel('Future timestep')
            ax.set_ylabel('Expected reward')
            plt.tight_layout()
            plt.savefig(f"../explanations/trd-natural-language/trd-obs-{explain_obs.id}-agent-{agent_id}.png")
            # plt.show()
            plt.close(fig)

    generated_date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    with open(f'../explanations/trd-natural-language/prompt-{generated_date}.json', 'w') as file:
        json.dump(
            {
                "explanations": explanations,
                "prompt": no_domain_knowledge_prompt
            },
            file
        )
    with open(f'../explanations/trd-natural-language/reward-summarises.json', 'w') as file:
        json.dump(explanations, file)
