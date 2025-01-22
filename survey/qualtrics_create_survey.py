import json
import pprint

import requests

api_token = "ADD_YOUR_OWN"
survey_id = "ADD_YOUR_OWN"
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "X-API-TOKEN": api_token
}


mechanism_to_id = {
    "DatasetSimilarity": "D",
    "TemporalDecomposition": "T",
    "SARFA": "S",
    "OptimalAction": "A"
}

strategy_to_id = {
    "Dots": "D",
    "EnergyPill": "E",
    "Survival": "S",
    "LoseALife": "L"
}

qualtric_file_ids = {}
with open("qualtric_file_ids.txt") as file:
    for line in file.readlines():
        filename, qualtric_id = line.strip().split("	")
        qualtric_file_ids[filename] = qualtric_id


with open("../explanations/optimal_action_explanation.json") as file:
    obs_optimal_action = json.load(file)


def add_strategy_id_block(obs_id: int, explanation_mechanism: str, agent_strategy: str):
    if explanation_mechanism == "DatasetSimilarity":
        explanation = f'<img style="width: 160px; height: 210px;" src="https://southampton.qualtrics.com/CP/Graphic.php?IM={qualtric_file_ids[f"dse-obs-{obs_id}-agent-{agent_strategy}-v2.gif"]}">'
    elif explanation_mechanism == "TemporalDecomposition":
        explanation = f'<img style="width: 500px; height: 250px;" src="https://southampton.qualtrics.com/CP/Graphic.php?IM={qualtric_file_ids[f"trd-obs-{obs_id}-agent-{agent_strategy}-v2.png"]}">'
    elif explanation_mechanism == "SARFA":
        explanation = f'<img style="width: 160px; height: 210px;" src="https://southampton.qualtrics.com/CP/Graphic.php?IM={qualtric_file_ids[f"sarfa-obs-{obs_id}-agent-{agent_strategy}-v2.png"]}">'
    elif explanation_mechanism == "OptimalAction":
        explanation = f'<span style="font-size:13px;">{obs_optimal_action[f"obs-{obs_id}-{agent_strategy}-v2"]}</span>'
    else:
        raise ValueError(f"unknown explanation mechanism, {explanation_mechanism}")

    question_table = f"""

    <table style="width:500px;" cellspacing="1" cellpadding="1" border="1">
        <tbody>
        <tr>
            <td>Context<br><img style="width: 160px; height: 210px;" src="https://southampton.qualtrics.com/CP/Graphic.php?IM={qualtric_file_ids[f'obs-{obs_id}.png']}"></td>

            <td>Explanation<br>{explanation}</td>
        </tr>
        </tbody>
    </table>"""

    # create strategy identification
    strategy_id_question_response = requests.post(
        url=f"https://fra1.qualtrics.com/API/v3/survey-definitions/{survey_id}/questions",
        headers=headers,
        json={
            "QuestionText": f"What strategy best describes what the explanation shows for this context? {question_table}",
            "ChoiceOrder": ["1", "2", "3", "4"],
            "Choices": {
                "1": {"Display": "Eat all the dots"},
                "2": {"Display": "Survival, don't get eaten by the ghosts"},
                "3": {"Display": "Lose a Life, be eaten by the ghosts"},
                "4": {"Display": "Eat the energy pills and ghosts"},
            },
            "DataExportTag": f"Obs-{obs_id}-{explanation_mechanism}-{agent_strategy}",
            "Language": [],
            "QuestionType": "MC",
            "Randomization": {"Type": "All"},
            "Selector": "SAHR",
            "SubSelector": "TX",
            "Validation": {"Settings": {
                "ForceResponse": "ON",
                "ForceResponseType": "ON",
                "Type": "None"
            }}
        }
    ).json()
    assert strategy_id_question_response["meta"][
               "httpStatus"] == "200 - OK", f'{explanation_mechanism=}, {agent_strategy=}, {obs_id=}, {strategy_id_question_response=}'
    strategy_question_id = strategy_id_question_response["result"]["QuestionID"]

    # create confidence question
    confidence_question_response = requests.post(
        url=f"https://fra1.qualtrics.com/API/v3/survey-definitions/{survey_id}/questions",
        headers=headers,
        json={
            "ChoiceOrder": ["1", "2", "3", "4", "5"],
            "Choices": {
                "1": {"Display": "Very Unconfident"},
                "2": {"Display": "Unconfident"},
                "3": {"Display": "Neutral"},
                "4": {"Display": "Confident"},
                "5": {"Display": "Very Confident"},
            },
            "DataExportTag": f"Obs-{obs_id}-{explanation_mechanism}-{agent_strategy}-confidence",
            "Language": [],
            "QuestionText": "How confident are you?",
            "QuestionType": "MC",
            "Selector": "SAHR",
            "SubSelector": "TX",
            "Validation": {"Settings": {
                "ForceResponse": "ON",
                "ForceResponseType": "ON",
                "Type": "None"
            }}
        }
    ).json()
    assert confidence_question_response["meta"][
               "httpStatus"] == "200 - OK", f'{explanation_mechanism=}, {agent_strategy=}, {obs_id=}, {confidence_question_response=}'
    confidence_question_id = confidence_question_response["result"]["QuestionID"]

    # create time question
    time_question_response = requests.post(
        url=f"https://fra1.qualtrics.com/API/v3/survey-definitions/{survey_id}/questions",
        headers=headers,
        json={
            "ChoiceOrder": ["1", "2", "3", "4"],
            "Choices": {
                '1': {'Display': 'First Click'},
                '2': {'Display': 'Last Click'},
                '3': {'Display': 'Page Submit'},
                '4': {'Display': 'Click Count'}
            },
            # 'Configuration': {'MaxSeconds': '0', 'MinSeconds': '0', 'QuestionDescriptionOption': 'UseText'},
            "DataExportTag": f"Obs-{obs_id}-{explanation_mechanism}-{agent_strategy}-timer",
            'DataVisibility': {'Hidden': False, 'Private': False},
            'DefaultChoices': False,
            'GradingData': [],
            "Language": [],
            "QuestionText": "Timing",
            "QuestionDescription": "Timing",
            "QuestionType": "Timing",
            "Selector": "PageTimer",
        }
    ).json()
    assert time_question_response["meta"]["httpStatus"] == "200 - OK", f'{explanation_mechanism=}, {agent_strategy=}, {obs_id=}, {time_question_response=}'
    time_question_id = time_question_response["result"]["QuestionID"]

    # questions block
    create_block_response = requests.post(
        url=f"https://fra1.qualtrics.com/API/v3/survey-definitions/{survey_id}/blocks",
        headers=headers,
        json={
            "Type": "Standard",
            "Description": f"Strategy ID - Obs: {obs_id}, Explanation: {explanation_mechanism}, Strategy: {agent_strategy}",
            "BlockElements": [],  # has to be achieved through a block update not create
        }
    ).json()
    assert create_block_response["meta"][
               "httpStatus"] == "200 - OK", f'{explanation_mechanism=}, {agent_strategy=}, {obs_id=}, {create_block_response=}'
    block_id = create_block_response["result"]["BlockID"]
    flow_id = create_block_response["result"]["FlowID"]

    update_block_response = requests.put(
        url=f"https://fra1.qualtrics.com/API/v3/survey-definitions/{survey_id}/blocks/{block_id}",
        headers=headers,
        json={
            'Type': 'Standard',
            'SubType': '',
            'Description': f"Strategy ID - Obs: {obs_id}, Explanation: {explanation_mechanism}, Strategy: {agent_strategy}",
            'ID': block_id,
            'Options': {"BlockLocking": "false", "RandomizeQuestions": 'false',
                        'BlockVisibility': 'Expanded'},
            'BlockElements': [
                {'Type': "Question", "QuestionID": strategy_question_id},
                {"Type": "Question", "QuestionID": confidence_question_id},
                {"Type": "Question", "QuestionID": time_question_id},
            ]
        }
    ).json()
    assert update_block_response["meta"]["httpStatus"] == "200 - OK", f'{explanation_mechanism=}, {agent_strategy=}, {obs_id=}, {update_block_response=}'

    return block_id, flow_id


def add_overall_explanation_block(explanation_mechanism):
    # create summary confidence question
    overall_confidence_question_response = requests.post(
        url=f"https://fra1.qualtrics.com/API/v3/survey-definitions/{survey_id}/questions",
        headers=headers,
        json={
            "QuestionText": "What was your overall confident using the explanations to prediction of the agent's strategy?",
            "ChoiceOrder": ["1", "2", "3", "4", "5"],
            "Choices": {
                "1": {"Display": "Very Unconfident"},
                "2": {"Display": "Unconfident"},
                "3": {"Display": "Neutral"},
                "4": {"Display": "Confident"},
                "5": {"Display": "Very Confident"},
            },
            "DataExportTag": f"{explanation_mechanism}-confidence",
            "Language": [],
            "QuestionType": "MC",
            "Selector": "SAHR",
            "SubSelector": "TX",
            "Validation": {"Settings": {
                "ForceResponse": "ON",
                "ForceResponseType": "ON",
                "Type": "None"
            }}
        }
    ).json()
    assert overall_confidence_question_response["meta"]["httpStatus"] == "200 - OK", f'{explanation_mechanism=}, {overall_confidence_question_response=}'
    overall_confidence_question_id = overall_confidence_question_response["result"]["QuestionID"]

    # create summary ease question
    overall_ease_question_response = requests.post(
        url=f"https://fra1.qualtrics.com/API/v3/survey-definitions/{survey_id}/questions",
        headers=headers,
        json={
            "QuestionText": "How easy was it to identify the agent's strategy based on the explanation provided?",
            "ChoiceOrder": ["1", "2", "3", "4", "5"],
            "Choices": {
                "1": {"Display": "Very Difficult"},
                "2": {"Display": "Somewhat Difficult"},
                "3": {"Display": "Neutral"},
                "4": {"Display": "Somewhat Easy"},
                "5": {"Display": "Very Easy"},
            },
            "DataExportTag": f"{explanation_mechanism}-ease",
            "Language": [],
            "QuestionType": "MC",
            "Selector": "SAHR",
            "SubSelector": "TX",
            "Validation": {"Settings": {
                "ForceResponse": "ON",
                "ForceResponseType": "ON",
                "Type": "None"
            }}
        }
    ).json()
    assert overall_ease_question_response["meta"]["httpStatus"] == "200 - OK", f'{explanation_mechanism=}, {overall_ease_question_response=}'
    overall_ease_question_id = overall_ease_question_response["result"]["QuestionID"]

    # create summary understanding question
    overall_understanding_question_response = requests.post(
        url=f"https://fra1.qualtrics.com/API/v3/survey-definitions/{survey_id}/questions",
        headers=headers,
        json={
            "QuestionText": "How well did the explanation help you understand the agent's strategy?",
            "ChoiceOrder": ["1", "2", "3", "4", "5"],
            "Choices": {
                "1": {"Display": "Did not understand at all"},
                "2": {"Display": "Understood very little"},
                "3": {"Display": "Neutral"},
                "4": {"Display": "Mostly understood"},
                "5": {"Display": "Completely understood"},
            },
            "DataExportTag": f"{explanation_mechanism}-understanding",
            "Language": [],
            "QuestionType": "MC",
            "Selector": "SAHR",
            "SubSelector": "TX",
            "Validation": {"Settings": {
                "ForceResponse": "ON",
                "ForceResponseType": "ON",
                "Type": "None"
            }}
        }
    ).json()
    assert overall_understanding_question_response["meta"]["httpStatus"] == "200 - OK", f'{explanation_mechanism=}, {overall_understanding_question_response=}'
    overall_understanding_question_id = overall_understanding_question_response["result"]["QuestionID"]

    # text box
    overall_textbox_question_response = requests.post(
        url=f"https://fra1.qualtrics.com/API/v3/survey-definitions/{survey_id}/questions",
        headers=headers,
        json={
            "QuestionType": "TE",
            "Selector": "ML",
            "DataExportTag": f"{explanation_mechanism}-additional-thoughts",
            "QuestionText": "Do you have any additional thoughts? (Optional)",
            "Language": [],
        }
    ).json()
    assert overall_textbox_question_response["meta"]["httpStatus"] == "200 - OK", f'{explanation_mechanism=}, {overall_textbox_question_response=}'
    overall_textbox_question_id = overall_textbox_question_response["result"]["QuestionID"]

    # questions block
    create_block_response = requests.post(
        url=f"https://fra1.qualtrics.com/API/v3/survey-definitions/{survey_id}/blocks",
        headers=headers,
        json={
            "Type": "Standard",
            "Description": f"Overall {explanation_mechanism} rating",
            "BlockElements": [],  # has to be achieved through a block update not create
        }
    ).json()
    assert create_block_response["meta"][
               "httpStatus"] == "200 - OK", f'{explanation_mechanism=}, {create_block_response=}'
    block_id = create_block_response["result"]["BlockID"]
    flow_id = create_block_response["result"]["FlowID"]

    update_block_response = requests.put(
        url=f"https://fra1.qualtrics.com/API/v3/survey-definitions/{survey_id}/blocks/{block_id}",
        headers=headers,
        json={
            'Type': 'Standard',
            'SubType': '',
            "Description": f"Overall {explanation_mechanism} rating",
            'ID': block_id,
            'Options': {"BlockLocking": "false", "RandomizeQuestions": 'false', 'BlockVisibility': 'Expanded'},
            'BlockElements': [
                {'Type': "Question", "QuestionID": overall_confidence_question_id},
                {"Type": "Question", "QuestionID": overall_ease_question_id},
                {"Type": "Question", "QuestionID": overall_understanding_question_id},
                {"Type": "Question", "QuestionID": overall_textbox_question_id},
            ]
        }
    ).json()
    assert update_block_response["meta"]["httpStatus"] == "200 - OK", f'{explanation_mechanism=}, {update_block_response=}'

    return block_id, flow_id


def create_all_blocks(number_of_obs: int):
    explanation_block_flow_ids = {}
    overall_block_flow_ids = {}

    for explanation_mechanism in mechanism_to_id:
        block_flow_ids = []
        for agent_strategy in strategy_to_id:
            for obs_id in range(number_of_obs):
                print(f'\tCreate {explanation_mechanism=}, {agent_strategy=}, {obs_id=} id block')

                block_flow_ids.append(add_strategy_id_block(obs_id, explanation_mechanism, agent_strategy))

        explanation_block_flow_ids[explanation_mechanism] = block_flow_ids

        print(f'Create overall {explanation_mechanism=} block')
        overall_block_flow_ids[explanation_mechanism] = add_overall_explanation_block(explanation_mechanism)

    with open("survey_block_flow_ids.json", "w") as file:
        json.dump({"explanations": explanation_block_flow_ids, "overall": overall_block_flow_ids}, file)


def update_flow(pos=1):
    flow_response = requests.get(
        url=f"https://fra1.qualtrics.com/API/v3/survey-definitions/{survey_id}/flow",
        headers=headers
    ).json()
    assert flow_response["meta"]["httpStatus"] == "200 - OK", f'{flow_response=}'

    flow_structure = flow_response["result"]
    # pprint.pprint(flow_structure)

    with open("survey_block_flow_ids.json") as file:
        survey_block_flow_ids = json.load(file)

    explanation_block_flow_ids = survey_block_flow_ids["explanations"]
    overall_block_flow_ids = survey_block_flow_ids["overall"]

    assert set(flow["Description"] for flow in flow_structure["Flow"][pos]["Flow"]) == set(mechanism_to_id.keys())
    for i, flow in enumerate(flow_structure["Flow"][pos]["Flow"]):
        explanation = flow["Description"]
        flow_structure["Flow"][pos]["Flow"][i]["Flow"][0]["Flow"] = [
            {
                "Autofill": [],
                "FlowID": flow_id,
                "ID": block_id,
                "Type": "Standard"
            }
            for block_id, flow_id in explanation_block_flow_ids[explanation]
        ]
        block_id, flow_id = overall_block_flow_ids[explanation]
        flow_structure["Flow"][pos]["Flow"][i]["Flow"].append(
            {
                "Autofill": [],
                "FlowID": flow_id,
                "ID": block_id,
                "Type": "Standard",
            }
        )
    flow_structure["Flow"] = flow_structure["Flow"][:pos+2]

    pprint.pprint(flow_structure)

    update_flow_response = requests.put(
        url=f"https://fra1.qualtrics.com/API/v3/survey-definitions/{survey_id}/flow",
        headers=headers,
        json=flow_structure
    ).json()
    print(f'{update_flow_response=}')


if __name__ == "__main__":
    # 1. Create the randomizer blocks for each explanation
    # 2. create all the question blocks
    create_all_blocks(number_of_obs=20)
    # 3. update the flows of the randomizer blocks to include all question
    update_flow(pos=1)
