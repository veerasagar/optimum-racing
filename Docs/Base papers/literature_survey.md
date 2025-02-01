# Literature Survey

| No. | Title                          | Author(s)         | Year | Summary                                      |
|-----|--------------------------------|-------------------|------|----------------------------------------------|
| 1   | Deep-Racing: An Embedded Deep Neural Network (EDNN) Model to Predict the Winning Strategy in Formula One Racing | Fatima, Syeda Sitara Wishal, and Jennifer Johrendt | 2023 | The paper introduces Deep-Racing, an EDNN model designed to predict driver rankings and optimal pitstop strategies in Formula One racing. |
| 2   | Example Paper Title 2          | Author C, Author D| 2022 | Brief summary of the paper                   |
| 3   | Example Paper Title 3          | Author E, Author F| 2021 | Brief summary of the paper                   |

## Paper 1

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

## Title: A Race Simulation for Strategy Decisions in Circuit Motorsports

- **cite:** Alexander Heilmeier, Michael Graf, and Markus Lienkarnp. 2018. A Race Simulation for Strategy Decisions in Circuit Motorsports. In 2018 21st International Conference on Intelligent Transportation Systems (ITSC). IEEE Press, 2986–2993. <https://doi.org/10.1109/ITSC.2018.8570012>
- **Summary:** This paper presents a race simulation tool designed to support strategy decisions in circuit motorsports, particularly in Formula 1. The simulation focuses on evaluating race strategies by modeling key factors such as tire degradation, fuel mass loss, pit stops, and overtaking maneuvers. The tool is based on a lap-wise discretization approach, which allows for quick and efficient simulation of entire races using publicly accessible lap time data.
- **Key Contributions:**

  - Race Simulation Framework:
    The simulation models the entire race, including long-term effects like tire wear and fuel consumption, as well as race events such as pit stops and overtaking. It uses empirical models to keep computational effort low, making it suitable for real-time strategy adjustments during races.

  - Lap-Wise Discretization:
    The race is simulated lap by lap, with each lap time calculated based on a base lap time (derived from qualifying or practice sessions) and adjusted for factors like tire degradation, fuel mass, and pit stops.This approach simplifies the simulation by avoiding the need for detailed track modeling or individual racing lines.

  - Tire Degradation Model:
    The simulation includes a logarithmic tire degradation model that accounts for the performance drop of tires over time. Different tire compounds (e.g., Ultrasoft, Supersoft) are modeled individually, with parameters adjusted for each driver or team.

  - Fuel Mass and Pit Stop Modeling:
    Fuel consumption is modeled linearly, with lap times increasing as fuel mass decreases.Pit stops are simulated with time penalties for inlaps and outlaps, and the simulation accounts for the dynamic standing times in Formula 1 (e.g., tire changes, penalties).

  - Overtaking Model:
    Overtaking is modeled based on time gaps between cars and the use of DRS (Drag Reduction System). The simulation includes a team order matrix to account for team strategies (e.g., favoring one driver over another).

  - Validation with Real Data:
    The simulation is validated using data from the 2017 Abu Dhabi Grand Prix. The results show a good match between simulated and actual race times, with deviations typically within 0.47 seconds on average.

- **Results:**
    The simulation accurately predicts race outcomes, including the effects of tire degradation, fuel consumption, and pit stops.It demonstrates the ability to support race engineers in pre-race strategy planning and real-time strategy adjustments during races.

## Paper 4

## Title: AI-enabled prediction of sim racing performance using telemetry data

- **cite:** Fazilat Hojaji, Adam J. Toth, John M. Joyce, Mark J. Campbell,
AI-enabled prediction of sim racing performance using telemetry data,
Computers in Human Behavior Reports,Volume 14,2024,100414,ISSN 2451-9588, <https://doi.org/10.1016/j.chbr.2024.100414>.
- **Summary:** The paper explores the application of data science and machine learning (ML) techniques to analyze and predict performance in sim racing, a rapidly growing segment of esports. The study focuses on identifying key in-game metrics that influence driving performance and provides insights into how these metrics can be used to improve training and performance in sim racing.
- **Applications:**

  - Training Tools: The identified metrics can be used to develop targeted training programs for sim racers.

  - Game Development: Insights from the study can help game developers enhance the realism and accuracy of racing simulators.

  - Strategy Development: Teams can use the findings to develop better racing strategies and optimize vehicle setups.

- **Results:**

  - Key Performance Metrics: The XGBoost model achieved the highest prediction accuracy (97.19%) and identified speed, lateral acceleration, and steering angle as the most influential metrics.

  - Driving Patterns: FAST laps were characterized by higher throttle application, more efficient braking, and smoother steering control compared to SLOW laps.

  - Track Difficulty: The analysis revealed that certain corners, particularly T2, were more challenging and had a significant impact on overall lap performance.
- **Future Research:**
    The study suggests expanding the analysis to other tracks and exploring additional ML techniques for trajectory prediction and driver behavior modeling. The methodology could also be applied to real-world driving scenarios, including autonomous vehicles. Overall, the paper demonstrates the potential of AI and ML in esports analytics, particularly in sim racing, and provides a framework for future research in this area.
