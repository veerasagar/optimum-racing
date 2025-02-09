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

## Paper 5

## Title: An Analysis of Time Series Analysis and Forecasting Techniques

- **cite:** Malik, Pankaj & Dangi, Aditya & Singh, Aditya & Asst, Thakur & Pratap, Aditya & Parihar, Singh & Sharma, Utkarsh & Mishra, Lakshya. (2023). An Analysis of Time Series Analysis and Forecasting Techniques. IJARCCE. Vol-9. 2023.
- **Year:** 2023
- **Link:** <https://www.researchgate.net/publication/375238697_An_Analysis_of_Time_Series_Analysis_and_Forecasting_Techniques>
- **Summary:** This research paper provides a comprehensive analysis of time series analysis and forecasting techniques, examining their efficacy, applicability, and interpretability. The paper explores both traditional statistical methods like ARIMA and Exponential Smoothing, as well as modern machine learning techniques like LSTM networks and ensemble methods
- **Key Points:**
  - Methodology: The paper outlines a systematic methodology for evaluating time series forecasting techniques, including dataset selection, evaluation metrics, experimental setup, and comparative analysis.
  - Comparative Analysis: The paper presents a detailed comparative analysis of the performance of various forecasting methods across different datasets and scenarios, highlighting their strengths, limitations, and adaptability to data characteristics.
  - Real-World Applications: The paper showcases real-world applications of time series forecasting in diverse domains, including demand forecasting, financial market predictions, disease outbreak prediction, and energy consumption forecasting.
  - Interpretability and Explainability: The paper emphasizes the importance of interpretability and explainability in time series forecasting, exploring techniques like SHAP values to provide insights into model decisions and enhance transparency.
  - Challenges and Future Directions: The paper discusses key challenges faced by practitioners and researchers, including handling non-stationarity, uncertainty quantification, and ethical considerations. It also outlines potential avenues for future research, such as developing hybrid models, integrating external data, and advancing explainable deep learning.
- **Conclusion:** The paper concludes that time series analysis and forecasting are essential tools for informed decision-making across various sectors. It provides a valuable resource for practitioners and researchers seeking to understand the nuances of different forecasting techniques and navigate the evolving landscape of this dynamic field.

## Paper 7

## Title: Application of Monte Carlo Methods to Consider Probabilistic Effects in a Race Simulation for Circuit Motorsport

- **cite:** Heilmeier, A.; Graf, M.; Betz, J.; Lienkamp, M. Application of Monte Carlo Methods to Consider Probabilistic Effects in a Race Simulation for Circuit Motorsport. Appl. Sci. 2020, 10, 4229. <https://doi.org/10.3390/app10124229>
- **Year:** 2020
- **Summary:** This paper presents a novel approach to modeling probabilistic effects in a race simulation for circuit motorsport, specifically focusing on Formula 1. The authors argue that traditional race simulations often fail to account for the significant impact of random events, such as accidents, safety car phases, and driver variability, which can drastically alter race outcomes.
- **Key Points:**
  - Modeling of Starting Performance: A driver-specific Gaussian distribution is used to model the variability in starting performance, based on a reference curve derived from real race data.
  - Modeling of Variability of Lap Time and Pit Stop Duration: Existing models for lap time and pit stop duration variability are adapted and parameterized using a larger database.
  - Determination of Accident and Failure Probabilities: Bayesian inference is employed to determine driver-specific accident probabilities and team-specific failure probabilities, taking into account historical data.
  - Determination of Full Course Yellow Phases: A novel approach is introduced to determine the quantity, start time, and duration of full course yellow (FCY) phases, considering their relationship with accidents and failures.
  - Modeling of Safety Cars: A "safety car ghost" (SCG) concept is proposed to realistically model the impact of safety cars on individual drivers, despite the lap-wise discretization of the simulation.
- **Conclusion:** The authors demonstrate the effectiveness of their approach through Monte Carlo simulations, showing that their model accurately captures the impact of probabilistic effects on race outcomes. They highlight the benefits of using Monte Carlo simulations for evaluating the robustness of different race strategies against unforeseen events.

## Paper 8

## Title: Bayesian analysis of Formula One race results: disentangling driver skill and constructor advantage

- **cite:** van Kesteren, Erik-Jan, and Tom Bergkamp. "Bayesian analysis of Formula One race results: disentangling driver skill and constructor advantage." Journal of quantitative analysis in sports 19.4 (2023): 273-293.
- **Year:** 2023
- **Summary:** This research paper investigates the relative contributions of driver skill and constructor advantage to success in Formula One racing. Using a novel Bayesian multilevel rank-ordered logit regression model, the authors analyze race finishing positions from the hybrid era (2014-2021).
- **Key Points:**
  - Driver Skill: Hamilton and Verstappen are identified as the most skilled drivers in the hybrid era.
  - Constructor Advantage: The top three teams (Mercedes, Ferrari, and Red Bull) consistently outperform other constructors.
  - Relative Importance: The model estimates that approximately 88% of the variance in race results is explained by the constructor, indicating that the car is more important than the driver in determining success.
- **Conclusion:** The approach has potential applications in other sports where multiple independent components contribute to success. The model provides a framework for quantifying and comparing driver skill and constructor advantage in Formula One.

## Paper 9

## Title: Deep Neural Network-based lap time forecasting of Formula 1 Racing

- **cite:** Zhao, Zhixuan. (2024). Deep Neural Network-based lap time forecasting of Formula 1 Racing. Applied and Computational Engineering. 47. 61-66. 10.54254/2755-2721/47/20241191.
- **Year:** 2024
- **Summary:** This paper proposes a Deep Neural Network (DNN) model for predicting the fastest lap time in Formula 1 qualifying sessions. The model leverages historical data from 2014 onwards, focusing on qualifying fastest lap times to minimize external factors.
- **Key Points:**
  - Data-Driven Approach: The DNN learns patterns from historical data to predict future performance, surpassing traditional methods like linear regression.
  - Driver and Track Specificity: The model considers individual driver performance at each circuit, enabling more accurate predictions.
  - Network Structure: A Fully Connected Neural Network (FCNN) with two hidden layers is employed, utilizing activation functions like PReLU and Tanh for efficient training.
  - Loss Function and Optimizer: SmoothL1Loss and Adam optimizer are used for parameter updates, ensuring fast convergence and adaptability.
- **Limitations:**
  - The model cannot account for external factors like weather conditions or driver status.
  - The Single Input Single Output (SISO) system limits the model's ability to consider complex interactions.
- **Future Research:**
Incorporating data from other drivers and races to improve prediction accuracy.
Developing a more comprehensive system that includes external factors for more realistic predictions.
Utilizing the model for player performance assessment and training.

## Paper 10

## Title: Evolutionary F1 Race Strategy

- **cite:** Andrea Bonomi, Evelyn Turri, and Giovanni Iacca. 2023. Evolutionary F1 Race Strategy. In Proceedings of the Companion Conference on Genetic and Evolutionary Computation (GECCO '23 Companion). Association for Computing Machinery, New York, NY, USA, 1925–1932. <https://doi.org/10.1145/3583133.3596349>
- **Year:** 2023
- **Summary:** This paper proposes a custom Genetic Algorithm (GA) to optimize Formula 1 race strategies. The GA considers various factors affecting race performance, including tire choice, pit stops, fuel weight, and tire wear. It simulates multiple race strategies and provides valuable insights for informed strategic decisions.
- **Key Points:**
  - Individual Representation: Each race strategy is represented as an individual with a genotype containing tire compound, fuel load, and pit stop information.
  - Fitness Function: The fitness value is the total race time, calculated by summing time losses due to tire compound, tire wear, fuel weight, pit stops, and weather conditions.
  - Selection and Replacement: The selection process uses a dynamic penalty function to prioritize strategies that are close to optimal, even if they violate constraints slightly. Elitism is used to preserve the best individuals from the previous generation.
  - Crossover: The crossover operation exchanges the initial fuel load between two strategies.
  - Mutation: Four types of mutations are implemented: tire compound change, pit stop removal, pit stop addition, and fuel load adjustment.
- **Conclusion:** The proposed GA demonstrates the effectiveness of Evolutionary Computation in optimizing F1 race strategies. Its speed and adaptability make it a valuable tool for real-time strategy analysis and decision-making.

## Paper 11

## Title: Explainable Reinforcement Learning for Formula One Race Strategy

- **cite:** Thomas, Devin, et al. "Explainable Reinforcement Learning for Formula One Race Strategy." arXiv preprint arXiv:2501.04068 (2025).
- **Year:** 2025
- **Summary:** This paper introduces RSRL (Race Strategy Reinforcement Learning), a reinforcement learning model designed to optimize race strategies in Formula One simulations. RSRL outperforms both a fixed strategy baseline and the industry standard Monte Carlo-based race strategy model provided by Mercedes-AMG PETRONAS F1 Team.
- **Key Points:**
  - Flexible and Portable Architecture: RSRL utilizes a flexible architecture that allows for training and deployment with different data sources, including live race data and simulations.
  - Improved Performance: RSRL achieves an average finishing position of P5.33 in the 2023 Bahrain Grand Prix, outperforming the best baseline model by 0.28 positions.
  - Generalisability Study: The paper demonstrates how RSRL can be trained to prioritize performance on specific tracks or across multiple tracks.
  - Explainable AI (XAI) Techniques: RSRL incorporates three XAI techniques – TimeSHAP, VIPER, and decision tree counterfactuals – to provide explanations for its decisions, enhancing user trust and understanding.
  - Real-World Applications: The paper provides illustrations that demonstrate how RSRL replicates real-world strategic decisions made by race strategists.
- **Evaluation:**
  - Model Performance: RSRL consistently outperforms baseline models in simulations, achieving a significant improvement in average finishing position.
  - Generalisability: RSRL models trained on a larger number of tracks exhibit better generalisability but may sacrifice performance on specific tracks.
  - Explanations: The XAI techniques employed in RSRL demonstrate high fidelity and comprehensibility, providing accurate and understandable explanations for the model's decisions.
- **Future Research:**
  - Cooperative Strategies: Exploring the use of RSRL to control multiple cars and develop cooperative strategies.
  - Driver-Specific Strategies: Fine-tuning RSRL to suggest personalized strategies based on individual driver styles.
  - Human-in-the-Loop Feedback: Incorporating human expertise into the training process to further improve RSRL's performance.

## Paper 12

## Title: Planning Formula One race strategies using discrete-event simulation

- **cite:** Bekker, James & Lotz, W. (2009). Planning Formula One race strategies using discrete-event simulation. Journal of the Operational Research Society. 60. 10.1057/palgrave.jors.2602626.
- **Year:** 2009
- **Summary:** This paper presents a discrete-event simulation model designed to assist Formula One racing teams in planning and evaluating their race strategies. The model simulates key on-track events, including car failures, passing maneuvers, and pit stops, to provide teams with a potential competitive advantage.
- **Key Points:**
  - Time-based approach: The model captures the mechanical complexities and physical interactions of a race car with its environment through a time-based approach, where all effects are translated into a net time for a car to travel between points on the track.
  - Passing maneuvers: The model simulates passing maneuvers based on the relative pace and position of cars, allowing for realistic overtaking scenarios.
  - Pit stops: The model incorporates pit stops, including the time required to travel down the pit lane and the operational delay time associated with refueling, tire changes, and other pit stop activities.
- **Conclusion:** The paper demonstrates the model's application by evaluating the race strategies employed by a specific team during the three validated races. The results suggest that the team could have achieved better results with different pit-stop strategies. The model can be used to plan strategies for future events by simulating various scenarios and identifying the most advantageous options.
- **Future Research:** Future work includes refining the model to incorporate changing weather conditions and the effects of major accidents. The ultimate goal is to develop a real-time model that can analyze race strategies and provide decision-makers with immediate insights to react to racing incidents.

## Paper 13

## Title: Rank position forecasting in car racing

- **cite:** Peng, Bo, et al. "Rank position forecasting in car racing." 2021 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE, 2021.
- **Year:** 2021
- **Summary:** This paper addresses the challenging problem of rank position forecasting in car racing, a domain characterized by high dynamics, uncertainty, and sparse data. Existing methods, including statistical models, machine learning regression models, and deep forecasting models, struggle to accurately predict rank positions due to the complex interplay of factors like driver skill, car configuration, racing strategies, and unpredictable events like crashes and mechanical failures.
The authors propose RankNet, a novel deep learning model that effectively tackles these challenges. RankNet leverages a decomposition approach, separating the modeling of pit stop events from the modeling of rank position. This strategy allows for more accurate predictions by addressing the uncertainty inherent in pit stop decisions.
- **Key Points:**
  - Cause-effect analysis of pit stops: The paper identifies and analyzes the factors influencing pit stops, including resource constraints, anomaly events, and race strategies.
  - Model decomposition: RankNet decomposes the forecasting problem into two sub-models: a PitModel that predicts future race status (pit stops and caution laps) and a RankModel that forecasts rank positions based on the predicted race status.
  - Deep learning model selection: The authors explore different deep learning models for the RankModel and find that models with weaker assumptions on global dependency structures perform best.
  - Domain knowledge-based optimizations: RankNet incorporates domain knowledge through feature engineering and optimization techniques, significantly improving forecasting performance.
  - Performance evaluation: RankNet demonstrates significant performance improvements over baselines, achieving more than 10% MAE improvement consistently.
- **RankNet's architecture:**
  - PitModel: A multilayer perceptron (MLP) network that predicts the lap number of the next pit stop.
  - RankModel: A stacked LSTM encoder-decoder network that predicts rank positions based on historical rank data and predicted race status.
- **Conclusion:** Overall, RankNet offers a promising solution for rank position forecasting in car racing, providing valuable insights for race analysis and strategy optimization.

## Paper 14

## Title: Time Series Analysis and Modeling to Forecast: a Survey

- **cite:** Dama, Fatoumata, and Christine Sinoquet. "Time series analysis and modeling to forecast: A survey." arXiv preprint arXiv:2104.00164 (2021).
- **Year:** 2021
- **Summary:** This survey provides a comprehensive overview of time series modeling and forecasting techniques, focusing on parametric models for predictive purposes. It covers a broad spectrum of models, from traditional linear models to more recent deep learning approaches.
- **Key Points:**
  Stationarity: A key concept in time series analysis, stationarity refers to the time-independence of a process's statistical properties. The survey discusses different types of stationarity (strong and weak) and methods for testing stationarity.
  Time Series Decomposition: Nonstationary time series are often decomposed into deterministic components (trend and seasonality) and a remaining stochastic component. The survey presents various decomposition schemes (additive, multiplicative, and mixed) and describes popular models for each scheme.
  Linear Models: The survey covers three major linear models: autoregressive (AR), moving average (MA), and autoregressive moving average (ARMA). It explains their autocorrelation function structures, parameter learning algorithms, and forecasting procedures.
  Nonlinear Models: The survey explores five categories of nonlinear models: polynomial autoregressive (PAR), functional-coefficient autoregressive (FAR), Markov switching autoregressive (MSAR), smooth transition autoregressive (STAR), and autoregressive conditional heteroscedasticity (ARCH).
  Deep Learning: The survey highlights the growing importance of deep learning in time series forecasting. It discusses various deep neural network architectures, including Multilayer Perceptrons (MLPs), Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), Convolutional Neural Networks (CNNs), and Transformers.
- **Conclusion:** This is the first comprehensive survey dedicated to forecasting in time series, covering the entire process flow from decomposition to forecasting.
It offers a unified presentation of decomposition frameworks, linear and nonlinear time series models, and the relationships between stationarity and linearity.
It provides in-depth knowledge while covering a broad range of models and forecasting methods, spanning from conventional approaches to recent deep learning adaptations.
It identifies new avenues for future research in time series modeling and forecasting.
- **Future Research:** Further exploration of deep learning models for time series forecasting, including the development of new architectures and techniques.
Investigation of multivariate time series forecasting, both point forecasting and probabilistic forecasting, to leverage dependencies across multiple variables.
Enhancement of conventional time series models through the integration of general machine learning techniques, such as ensemble-based strategies, penalized regression, and clustering.
Development of advanced bivariate process models that capture both short-term and long-term dependencies, incorporating event traces and actions triggered by human beings.
Addressing the issue of data obsolescence in time series modeling and forecasting.

## Paper 15

## Title: Time series forecasting model for non-stationary series pattern extraction using deep learning and GARCH modeling

- **cite:** Han, Huimin & Liu, Zehua & Barrios, Mauricio & Li, Jiuhao & Zeng, Zhixiong & Sarhan, Nadia & Awwad, Emad. (2024). Time series forecasting model for non-stationary series pattern extraction using deep learning and GARCH modeling. Journal of Cloud Computing. 13. 10.1186/s13677-023-00576-7.
- **Year:** 2024
- **Summary:** This paper proposes a novel time series forecasting model that combines signal decomposition and deep learning techniques to address the challenges of non-linear and non-stationary time series data. The model utilizes the Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model to learn the volatility in time series changes, followed by Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN) for data decomposition. Finally, Graph Convolutional Networks (GCN) are applied to learn the features of the decomposed data.
- **Key Points:** The model is evaluated on three datasets: Air Quality, Energy, and Traffic. The results demonstrate that the proposed model outperforms traditional methods, particularly on the Energy and Traffic datasets. The model's strengths lie in its ability to capture dynamic volatility, reduce data complexity through decomposition, and effectively learn data relationships using GCN.
- **Conclusion:** The paper highlights the potential of this hybrid model for applications in various fields, including finance, energy, and retail, where accurate forecasting of non-stationary time series data is crucial. However, the model's complexity, computational demands, and potential for overfitting are acknowledged as limitations. Future research should focus on enhancing interpretability, reducing computational overhead, and exploring adaptability to different types of non-stationary data.

## Paper 16

## Title: Virtual Strategy Engineer: Using Artificial Neural Networks for Making Race Strategy Decisions in Circuit Motorsport

- **cite:** Heilmeier, A.; Thomaser, A.; Graf, M.; Betz, J. Virtual Strategy Engineer: Using Artificial Neural Networks for Making Race Strategy Decisions in Circuit Motorsport. Appl. Sci. 2020, 10, 7805. <https://doi.org/10.3390/app10217805>
- **Year:** 2020
- **Summary:** This paper presents a methodology for automating race strategy decisions in circuit motorsport, specifically focusing on Formula 1. The authors propose a Virtual Strategy Engineer (VSE) based on two artificial neural networks (NNs) to determine pit stop timing and tire compound selection.
- **Key Points:** Performance Evaluation: The VSE demonstrates reasonable decision-making, adapting to race situations like FCY phases and undercut attempts.
Feature Impact Analysis: The authors analyze the influence of individual features on the VSE's predictions, highlighting the importance of factors like tire age, FCY status, and remaining pit stops.
Comparison with Real-World Strategies: The VSE's performance is compared to real-world race strategies, showing potential for improving race outcomes.
- **Conclusion:** The VSE demonstrates the potential of artificial intelligence for automating race strategy decisions in circuit motorsport. Its ability to learn from real-world data and adapt to dynamic race situations makes it a valuable tool for improving race simulation realism and supporting real strategy engineers.

## Paper 17

## Title: Online Planning for F1 Race Strategy Identification

- **cite:** Piccinotti, D. I. E. G. O., et al. "„Online Planning for F1 Race Strategy Identification “." International Conference on Automated Planning and Scheduling (ICAPS). 2021.
- **Year:** 2021
- **Summary:** This paper investigates the use of online planning algorithms for identifying optimal race strategies in Formula 1 (F1). The authors model the race strategy problem as a Markov Decision Process (MDP) and propose an open-loop approach using Monte Carlo Tree Search (MCTS) with Temporal Difference (TD) updates.
- **Key Points:**
  - Open-Loop Planning: The authors employ an open-loop planning strategy to address the challenges of large continuous state spaces and stochastic transitions in the F1 environment. This approach simplifies the search tree by focusing on sequences of actions rather than state-action mappings.
  - Q-Learning TD Updates: To mitigate the high variance of returns in the search tree, the authors incorporate Q-learning TD updates into the MCTS algorithm. This helps to improve the stability and accuracy of value estimates.
  - Simulation Environment: The paper utilizes a modified version of a previously developed F1 race simulator to evaluate the proposed planning algorithms. The simulator incorporates probabilistic race events, driver performance models, and tire degradation factors.
  - Experimental Results: The authors conduct experiments on a set of F1 races from 2015 to 2018, comparing the performance of their Q-learning OL UCT algorithm with other planning methods and baselines. The results demonstrate that the proposed approach can improve race times and final positions compared to real-world strategies.
- **Conclusion:** The paper concludes that online planning algorithms can be a valuable tool for F1 strategists, providing real-time recommendations and potentially improving race outcomes.
- **Future Research:**
  - The performance of the planners is heavily influenced by the rollout policies used.
  - The reward function used in the experiments focuses on minimizing race time, which may not fully capture the complexities of F1 strategy.
  - Future work includes exploring the use of function approximators and multi-agent frameworks to enhance the planning capabilities.

## Paper 18

## Title: On the Optimization of Pit Stop Strategies via Dynamic Programming

- **cite:** Heine, O.F.C., Thraves, C. On the optimization of pit stop strategies via dynamic programming. Cent Eur J Oper Res 31, 239–268 (2023). <https://doi.org/10.1007/s10100-022-00806-4>
- **Year:** 2023
- **Summary:** This paper presents two dynamic programming models for optimizing pit stop strategies in Formula 1 races: a deterministic model and a stochastic model.

- **Deterministic Model:**
  - This model assumes no uncertain events and focuses on minimizing race time by optimizing tire compound choices and pit stop timing.
  - The model considers factors like tire wear, fuel consumption, and lap time variations based on tire compound and wear.
  - The model is solved using a Bellman equation and a border condition that ensures the car uses at least two different tire compounds during the race.
  - The model can be adapted to incorporate yellow flag events by re-solving the dynamic program from the lap where the yellow flag occurs.

- **Stochastic Model:**
  - This model extends the deterministic model by incorporating uncertainty in the form of weather changes and yellow flag events.
  - The model considers the probability of yellow flag occurrences and their duration, as well as the probability of weather transitions.
  - The model is solved using a Bellman equation and a border condition that accounts for the possibility of using wet tires, which suspends the requirement of using at least two different tire compounds.
  - The model demonstrates that delaying pit stops to potentially benefit from a yellow flag can lead to faster race times, especially when yellow flags are more likely to occur.
- **Key Points:**
  - The stochastic model outperforms the deterministic model in scenarios where yellow flags are more likely to occur.
  - The stochastic model tends to delay pit stops to potentially benefit from yellow flags, while the deterministic model makes pit stops based on pre-determined optimal timing.
  - The models can be applied to other motorsports and can be adapted to different functional forms for tire wear, fuel consumption, and lap time calculations.

- **Future Research:**
  - The models do not consider competition between drivers, which can significantly impact race strategy.
  - The model simplifies yellow flags as Virtual Safety Car (VSC) events that occur between the start and end of laps.
  - Incorporating driver competition and game theory aspects into the models.
  - Expanding the model to account for different types of yellow flag events and their impact on race strategy.

## Paper 19

## Title: Optimizing Pit Stop Strategies with Competition in a Zero-Sum Feedback Stackelberg Game in Formula 1

- **cite:**
- **Year:**
- **Summary:** This paper presents a game theory model for optimizing pit stop strategies in Formula 1 races, considering competition between two drivers. The model is formulated as a zero-sum feedback Stackelberg game using dynamic programming, where each driver decides whether to pit or stay on track at each lap.
- **Key Points:**
  - Competition: The model explicitly accounts for the interaction between drivers, including overtaking and defending positions, which affects their lap times.
  - Uncertainty: The model incorporates stochastic events like yellow flags (Virtual Safety Car and Safety Car) and randomness in lap times.
  - Tire Degradation: The model considers tire wear and the trade-off between using softer tires for faster lap times but shorter stints, and harder tires for longer stints but slower lap times.
  - Objective Functions: The model allows for different objective functions, including maximizing the time gap with the opponent and maximizing the probability of winning.
- **Conclusion:** The paper concludes by discussing potential extensions of the model, such as considering competition with more than two drivers and applying the model to other applications beyond Formula 1.
- **Main Contribution:**
  - Equilibrium Definition and Existence: The paper defines the feedback Stackelberg equilibrium and the Nash-feedback Stackelberg equilibrium, which include both pit stop strategies and initial tire compound decisions. The existence of these equilibria is proven.
  - Algorithm for Finding Equilibrium: A backward induction algorithm with dynamic programming is proposed to find the game equilibrium.
  - Numerical Results: The model is implemented and solved for instances with hundreds of millions of states. The results show that strategic decision-making significantly enhances players' race outcomes.

## Paper 20

## Title:

- **cite:**
- **Year:**
- **Summary:**
- **Key Points:**
- **Conclusion:**
- **Future Research:**
