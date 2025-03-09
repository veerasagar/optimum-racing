import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Define the new state representation using selected CSV columns
class RaceState:
    def __init__(self, row):
        # Use a subset of features as the state:
        # LapNumber, Sector1Time, Sector2Time, Sector3Time, TyreLife, Compound (encoded),
        # Position, TimeGapToLeader, TimeGapToBehind, AirTemp, TrackTemp
        self.lap_number = row["LapNumber"]
        self.sector1 = row["Sector1Time"]
        self.sector2 = row["Sector2Time"]
        self.sector3 = row["Sector3Time"]
        self.tyre_life = row["TyreLife"]
        # Encode Compound: HARD -> 0, MEDIUM -> 1 (extend as needed)
        self.compound = 0 if row["Compound"].strip().upper() == "HARD" else 1
        self.position = row["Position"]
        self.time_gap_leader = row["TimeGapToLeader"]
        self.time_gap_behind = row["TimeGapToBehind"]
        self.air_temp = row["AirTemp"]
        self.track_temp = row["TrackTemp"]

    def to_array(self):
        return np.array([
            self.lap_number, self.sector1, self.sector2, self.sector3,
            self.tyre_life, self.compound, self.position,
            self.time_gap_leader, self.time_gap_behind,
            self.air_temp, self.track_temp
        ])

# Define a simple binary action space: NO_PIT and PIT_STOP
class RaceAction:
    NO_PIT = 0
    PIT_STOP = 1

    @staticmethod
    def get_action_space():
        return [RaceAction.NO_PIT, RaceAction.PIT_STOP]

# Define the environment that steps through the CSV lap data
class RaceEnvironment:
    def __init__(self, filename):
        # Load the dataset (each row corresponds to one lap for one driver)
        self.data = pd.read_csv(filename)
        self.total_laps = len(self.data)
        self.current_index = 0
        self.state = None

    def reset(self):
        self.current_index = 0
        self.state = RaceState(self.data.iloc[self.current_index])
        return self.state

    def step(self, action):
        # Get the current lap row
        row = self.data.iloc[self.current_index]
        # Retrieve lap time and pit time (if any)
        lap_time = row["LapTime"]
        pit_time = row["PitTime"] if pd.notna(row["PitTime"]) else 0.0

        # Apply the agent’s decision:
        # If PIT_STOP is chosen, add the pit stop penalty (pit_time) to the lap time.
        effective_lap_time = lap_time + pit_time if action == RaceAction.PIT_STOP else lap_time
        reward = -effective_lap_time  # Lower lap times yield a higher (less negative) reward

        # Advance to the next lap
        self.current_index += 1
        done = (self.current_index >= self.total_laps)
        if not done:
            self.state = RaceState(self.data.iloc[self.current_index])
        else:
            self.state = None
        return self.state, reward, done

# Build the DRQN model (unchanged except for input shape)
def build_drqn_model(input_shape, action_space_size):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_space_size, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Training loop for the dataset-driven environment
def train_rsrl_model(env, episodes, gamma, epsilon, epsilon_decay, min_epsilon):
    # Determine the input shape from a sample state vector
    sample_state = env.reset().to_array()
    input_shape = (1, sample_state.size)  # (timesteps, features)
    action_space_size = len(RaceAction.get_action_space())
    model = build_drqn_model(input_shape, action_space_size)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(RaceAction.get_action_space())
            else:
                q_vals = model.predict(state.to_array().reshape(1, 1, -1), verbose=0)
                action = np.argmax(q_vals)

            next_state, reward, done = env.step(action)
            target = reward
            if not done:
                q_next = model.predict(next_state.to_array().reshape(1, 1, -1), verbose=0)
                target = reward + gamma * np.amax(q_next)

            # Update Q-value for the chosen action
            target_f = model.predict(state.to_array().reshape(1, 1, -1), verbose=0)
            target_f = target_f.reshape(-1)
            target_f[action] = target
            target_f = target_f.reshape(1, -1)

            model.fit(state.to_array().reshape(1, 1, -1), target_f, epochs=1, verbose=0)
            state = next_state
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

# Example usage:
filename = "ver_Monza_2024_laps.csv"  # update with your CSV filename
env = RaceEnvironment(filename)
episodes = 10
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

train_rsrl_model(env, episodes, gamma, epsilon, epsilon_decay, min_epsilon)
