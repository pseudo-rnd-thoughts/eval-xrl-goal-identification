# Evaluation of Explanations for Strategy Identification

This project involved implementing a strategy identification evaluation methodology for explainable reinforcement learning. 

The associated write up of this research will be available upon publication of my PhD or please email me for an early copy.

We use the evaluation methodology set out in the figure below
![Flowchart of the evaluation methodology for a single question](/figs/goal-identification-task.png)

This is part of a wide survey that asks self-reported subjective questions as well, see figure below
![Survey flowchart](/figs/comparative-user-evaluation.png)

I implement the following algorithms for the survey: TRD Summarisation (novel extension to [TRD](https://github.com/pseudo-rnd-thoughts/temporal-reward-decomposition) using GPT4o to generate natural language summarises of future expected rewards), [Dataset Similarity Explanation](https://github.com/pseudo-rnd-thoughts/temporal-explanations-4-drl), [SARFA](https://arxiv.org/abs/1912.12191) and Optimal Action Description (a natural language description of the policy's next action). See `explanation_mechanisms` implementations of each. 

The survey creation, data extraction, raw anonymised data and analyse code is available in `survey`. The used explanations in the survey are provided in `explanations`  

