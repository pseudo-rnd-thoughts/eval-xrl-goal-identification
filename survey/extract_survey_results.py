from typing import NamedTuple

import numpy as np
import pandas as pd

mechanisms = [
    "DatasetSimilarity",
    "TemporalDecomposition",
    "SARFA",
    "OptimalAction",
]

flow_to_mechanisms = {  # converts the flow id to mechanism name
    "FL_5": "DatasetSimilarity",
    "FL_6": "TemporalDecomposition",
    "FL_7": "SARFA",
    "FL_8": "OptimalAction",
}
flow_to_mechanism_flow = {  # converts the flow id to the mechanisms randomizer flow id
    "FL_5": "FL_255",  # dataset similarity
    "FL_6": "FL_173",  # temporal decomposition
    "FL_7": "FL_91",  # sarfa
    "FL_8": "FL_9",  # optimal action
}
strategy_name_to_strategy = {
    "Eat all the dots": "Dots",
    "Eat the energy pills and ghosts": "EnergyPill",
    "Survival, don't get eaten by the ghosts": "Survival",
    "Lose a Life, be eaten by the ghosts": "LoseALife",
}

assert set(flow_to_mechanisms.values()) == set(mechanisms)


class ParticipantInfo(NamedTuple):
    user_id: int
    gender: str  # categorical
    age_group: str  # categorical
    education: str  # categorical
    ai_study: str  # yes, no
    pacman_knowledge: str  # categorical
    time_taken: int  # time taken for the whole survey to be complete
    explanation_mechanism_order: tuple[str, ...]  # order of the explanation mechanisms shown to users
    final_thoughts: str


class StrategyIdAnswer(NamedTuple):
    user_id: int

    obs_id: int  # observation number
    explanation_mechanism: str  # explanation mechanism used
    true_strategy: str  # true agent strategy

    predicted_strategy: str  # predicted agent strategy
    is_correct: bool  # If the user got the answer correct
    confidence: str  # user confidence in answer
    time_taken: float  # time taken to answer the question in seconds

    survey_question_pos: int  # What was the users


class MechanismOverallRatings(NamedTuple):
    user_id: int
    explanation_mechanism: str  # explanation mechanisms overall

    confidence: str  # overall confidence
    ease: str  # overall ease
    understanding: str  # overall understanding
    additional_thoughts: str  # optional additional thoughts

    survey_mechanism_pos: int


def breakdown_user_answers(index, data) -> tuple[ParticipantInfo, list[StrategyIdAnswer], list[MechanismOverallRatings]]:
    mechanism_order = tuple(flow_to_mechanisms[flow] for flow in data["FL_3_DO"].split("|"))  # DO = Display Order

    participant_info = ParticipantInfo(
        user_id=index,
        gender=data["user-gender"],
        age_group=data["user-age"],
        education=data["user-education"],
        ai_study=data["user-ai-study"],
        pacman_knowledge=data["user-pacman"],
        time_taken=int(data["Duration (in seconds)"]),
        explanation_mechanism_order=mechanism_order,
        final_thoughts=data["final-thoughts"],
    )

    mechanisms_overall_ratings = [
        MechanismOverallRatings(
            user_id=index,
            explanation_mechanism=mechanism,
            confidence=data[f"{mechanism}-confidence"],
            ease=data[f"{mechanism}-ease"],
            understanding=data[f"{mechanism}-understanding"],
            additional_thoughts=data[f"{mechanism}-additional-thoughts"],
            survey_mechanism_pos=mechanism_order.index(mechanism),
        )
        for mechanism in mechanisms
    ]

    question_order = "|".join(data[f'{flow_to_mechanism_flow[flow]}_DO']
                              for flow in data["FL_3_DO"].split("|")).split("|")
    strategy_id_answers = []
    for strategy_id in list(filter(lambda col: "Obs" in col, data.dropna().index))[::7]:
        _, obs_id, mechanism, strategy = strategy_id.split("-")
        strategy_id_answers.append(StrategyIdAnswer(
            user_id=index,
            obs_id=obs_id,
            explanation_mechanism=mechanism,
            true_strategy=strategy,
            predicted_strategy=strategy_name_to_strategy[data[f'{strategy_id}']],
            is_correct=bool(strategy == strategy_name_to_strategy[data[f'{strategy_id}']]),
            confidence=data[f"{strategy_id}-confidence"],
            time_taken=data[f"{strategy_id}-timer_Page Submit"],
            survey_question_pos=question_order.index(f'StrategyID-Obs:{obs_id},Explanation:{mechanism},Strategy:{strategy}')
        ))

    return participant_info, strategy_id_answers, mechanisms_overall_ratings


if __name__ == "__main__":
    filename = "RAW_DATA"
    survey = pd.read_csv(filename)
    # print(f'{list(survey.columns)=}')

    all_participant_info, all_strategy_id_answers, all_overall_answers = [], [], []
    for row_num, row in survey.iterrows():
        if row["Finished"] == "True" and row["Status"] == "IP Address" and row["Progress"] == "100":
            # print(f'{row["attention-goal"]=}, {row["attention-strategy"]=}')
            if row["attention-goal"] == "To select the strategy that aligns with the explanation" or row["attention-strategy"] == "Lose a life":
                if row["PROLIFIC_PID"] != "673d6b4fb7181b12310a27e2":
                    user_answers = breakdown_user_answers(row_num, row)
                    assert len(user_answers[1]) == 16, len(user_answers[1])
                    assert len(user_answers[2]) == 4, len(user_answers[2])
                    all_participant_info.append(user_answers[0])
                    all_strategy_id_answers += user_answers[1]
                    all_overall_answers += user_answers[2]

                    print(f'[+] {row["PROLIFIC_PID"]} time taken={user_answers[0].time_taken // 60}m {user_answers[0].time_taken % 60}s, '
                          f'accuracy={np.mean([x.is_correct for x in user_answers[1]])}')
                else:
                    print("[-] Excluding 673d6b4fb7181b12310a27e2 as user was not paid.")
            else:
                print(f'[-] {row["PROLIFIC_PID"]} failed comprehension')
        else:
            print(f'[-] Initial fail ProlificID={row["PROLIFIC_PID"]}, Finished={row["Finished"]}, Status={row["Status"]}, Progress={row["Progress"]}')

    print(f'Summary - {len(all_participant_info)} participants, {len(all_strategy_id_answers)} strategy id answers, {len(all_overall_answers)} overall answers')

    pd.DataFrame(all_participant_info, columns=ParticipantInfo._fields).to_csv(
        f'{filename.replace(".csv", "")}-participants.csv', index=False)
    pd.DataFrame(all_strategy_id_answers, columns=StrategyIdAnswer._fields).to_csv(
        f'{filename.replace(".csv", "")}-strategy-id-answers.csv', index=False)
    pd.DataFrame(all_overall_answers, columns=MechanismOverallRatings._fields).to_csv(
        f'{filename.replace(".csv", "")}-mechanism-overall-ratings.csv', index=False)
