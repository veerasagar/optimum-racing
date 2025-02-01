# Literature Survey

| No. | Title                          | Author(s)         | Year | Summary                                      |
|-----|--------------------------------|-------------------|------|----------------------------------------------|
| 1   | Deep-Racing: An Embedded Deep Neural Network (EDNN) Model to Predict the Winning Strategy in Formula One Racing | Fatima, Syeda Sitara Wishal, and Jennifer Johrendt | 2023 | The paper introduces Deep-Racing, an EDNN model designed to predict driver rankings and optimal pitstop strategies in Formula One racing. |
| 2   | Example Paper Title 2          | Author C, Author D| 2022 | Brief summary of the paper                   |
| 3   | Example Paper Title 3          | Author E, Author F| 2021 | Brief summary of the paper                   |

## Paper 1

Author -- veersagar

## Title:  Deep-Racing: An Embedded Deep Neural Network (EDNN) Model to Predict the Winning Strategy in Formula One Racing

- **cite:**
    Fatima, Syeda Sitara Wishal, and Jennifer Johrendt. "Deep-Racing: An Embedded Deep Neural Network (EDNN) Model to Predict the Winning Strategy in Formula One Racing."

    link: <https://www.ijml.org/vol13/IJML-V13N3-1135-MT23-337.pdf>

- **Summary:**

    The paper introduces Deep-Racing, an Embedded Deep Neural Network (EDNN) model designed to predict driver rankings and optimal pitstop strategies in Formula One racing.

    The model uses data from Formula One seasons 2015-2022.

    It aims to assist team principals and race engineers.

    The authors suggest that further hyperparameter tuning and testing different embeddings could lead to even better results.

    It utilizes two separate neural networks, one for predicting the driver's rank and the other for determining the optimal lap for a pitstop.

    The model incorporates data preprocessing techniques, including data imputations and filtering to eliminate outliers and irrelevant data.

    Hyperparameters for the model are carefully selected, and performance is evaluated using metrics such as RMSE, R², precision, recall, and F1 score.

    The first Embedded Deep Neural Network (EDNN) model demonstrated an RMSE of 2.51 on the training dataset and 2.05 on the test dataset, with R2 scores of 0.42 and 0.39, respectively.

    The model's predictions showed a 93% correlation with actual driver ranks.

- **Dataset:**

    The dataset (<https://github.com/TUMFTM/f1-timing-database>) used in the study spans from 2015 to 2022 and is compiled from multiple resources.

    It includes data for 258 races, encompassing a total of 169,525 laps.

    Key race metrics such as race lap times, qualifying lap times, starting positions, pole positions, and pitstop durations were obtained from the Ergast API.

    The dataset also records the number of accidents and failures per driver and season from an online motorsport statistics site.

    Notably, the dataset reflects a data imbalance, with only 3.07% of laps involving a pitstop.

## Paper 2

## Title: Mastering Nordschleife - A comprehensive race simulation for AI strategy decision-making in motorsports

- **cite:** Boettinger, Max, and David Klotz. "Mastering Nordschleife--A comprehensive race simulation for AI strategy decision-making in motorsports." arXiv preprint arXiv:2306.16088 (2023).
- **Summary:** This paper presents a novel reinforcement learning (RL)-based approach to optimize race strategy decisions in GT motorsports, specifically for the Nürburgring Nordschleife circuit. The study addresses limitations in existing race simulations, which often require manual input and focus on F1 regulations, by developing a GT-specific model tailored to the Nürburgring Langstrecken Serie (NLS).
- **Objective:**
    Automate pit-stop and refueling decisions using RL, balancing fuel consumption, tire degradation, and time penalties.

    Create a realistic simulation environment for GT racing, incorporating sector-wise track discretization, probabilistic events (e.g., accidents, traffic), and NLS-specific rules like dynamic pit-stop standing times.

- **Methodology:**
    Simulation Design: Models a 4-hour NLS race (25 laps) with sector-based time adjustments for fuel, tire wear, and traffic. Historical 2020 NLS data validates parameters like tire degradation (logarithmic model) and fuel consumption (linear model).

    Reinforcement Learning: Implements a Deep Q-Network (DQN) with an action space (pit-stop refuel amounts) and observation space (fuel level, race position). Reward functions penalize high tire degradation and reward competitive positions.

    Hyperparameter Tuning: Evaluates learning rates, replay buffers, and episode counts, settling on 100,000 episodes for stable training.

- **Conclusion:**
    The simulation accurately replicates GT race dynamics, demonstrating RL’s potential for strategic automation. Challenges included sparse data for probabilistic events (e.g., multi-class traffic).

    Future work could explore self-play for robust policies and enhanced probabilistic modeling.

    This research bridges a gap in motorsport AI applications, offering tools for real-time decision-making in endurance racing beyond F1.

## Paper 3

## Title

- **cite:**
- **Summary:**
- **Dataset:**
