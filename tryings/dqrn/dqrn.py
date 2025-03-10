import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from collections import deque
import random

# Extended compound encoding for multiple compounds
compound_mapping = {
    "HARD": 0,
    "MEDIUM": 1,
    "SOFT": 2,
    # Add more mappings as needed
}

# Define the state representation using selected CSV columns, with basic missing value handling
class RaceState:
    def __init__(self, row):
        self.lap_number = row["LapNumber"] if pd.notna(row["LapNumber"]) else 0.0
        self.sector1 = row["Sector1Time"] if pd.notna(row["Sector1Time"]) else 0.0
        self.sector2 = row["Sector2Time"] if pd.notna(row["Sector2Time"]) else 0.0
        self.sector3 = row["Sector3Time"] if pd.notna(row["Sector3Time"]) else 0.0
        self.tyre_life = row["TyreLife"] if pd.notna(row["TyreLife"]) else 0.0
        compound_str = row["Compound"].strip().upper() if pd.notna(row["Compound"]) else "UNKNOWN"
        self.compound = compound_mapping.get(compound_str, -1)
        self.position = row["Position"] if pd.notna(row["Position"]) else 0.0
        self.time_gap_leader = row["TimeGapToLeader"] if pd.notna(row["TimeGapToLeader"]) else 0.0
        self.time_gap_behind = row["TimeGapToBehind"] if pd.notna(row["TimeGapToBehind"]) else 0.0
        self.air_temp = row["AirTemp"] if pd.notna(row["AirTemp"]) else 0.0
        self.track_temp = row["TrackTemp"] if pd.notna(row["TrackTemp"]) else 0.0

    def to_array(self):
        return np.array([
            self.lap_number, self.sector1, self.sector2, self.sector3,
            self.tyre_life, self.compound, self.position,
            self.time_gap_leader, self.time_gap_behind,
            self.air_temp, self.track_temp
        ], dtype=np.float32)

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
        self.data = pd.read_csv(filename)
        self.total_laps = len(self.data)
        self.current_index = 0
        self.state = None

    def reset(self):
        self.current_index = 0
        self.state = RaceState(self.data.iloc[self.current_index])
        return self.state

    def step(self, action):
        row = self.data.iloc[self.current_index]
        lap_time = row["LapTime"]
        pit_time = row["PitTime"] if pd.notna(row["PitTime"]) else 0.0
        # Apply pit penalty if PIT_STOP is chosen
        effective_lap_time = lap_time + pit_time if action == RaceAction.PIT_STOP else lap_time
        
        # Modified reward: Higher reward for lower lap times.
        # Assuming a constant offset (e.g., 120) to yield positive rewards.
        reward = 120 - effective_lap_time

        self.current_index += 1
        done = (self.current_index >= self.total_laps)
        if not done:
            self.state = RaceState(self.data.iloc[self.current_index])
        else:
            self.state = None
        return self.state, reward, done

# Build the DRQN model with an LSTM and dense layers
def build_drqn_model(input_shape, action_space_size):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_space_size, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Replay Buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Training loop with improvements: replay buffer, batch updates, and target network
def train_rsrl_model(env, episodes, gamma, epsilon, epsilon_decay, min_epsilon,
                     batch_size=32, replay_capacity=1000, target_update_freq=5):
    sample_state = env.reset().to_array()
    feature_size = sample_state.size
    input_shape = (1, feature_size)  # (timesteps, features)
    action_space_size = len(RaceAction.get_action_space())
    
    # Build the main and target networks
    model = build_drqn_model(input_shape, action_space_size)
    target_model = build_drqn_model(input_shape, action_space_size)
    target_model.set_weights(model.get_weights())

    replay_buffer = ReplayBuffer(replay_capacity)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_arr = state.to_array().reshape(1, 1, -1)
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(RaceAction.get_action_space())
            else:
                q_vals = model.predict(state_arr, verbose=0)
                action = np.argmax(q_vals[0])
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Store experience in replay buffer
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state

            # Perform a batch update if enough experiences are available
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                state_batch = np.array([s.to_array().reshape(1, -1) for s, a, r, s_next, done_flag in batch])
                state_batch = state_batch.reshape(batch_size, 1, feature_size)
                target_batch = model.predict(state_batch, verbose=0)

                # Update target for each sample in the mini-batch
                for i, (s, a, r, s_next, done_flag) in enumerate(batch):
                    if done_flag or s_next is None:
                        target_batch[i][a] = r
                    else:
                        s_next_arr = s_next.to_array().reshape(1, 1, -1)
                        t = target_model.predict(s_next_arr, verbose=0)
                        target_batch[i][a] = r + gamma * np.amax(t[0])
                model.fit(state_batch, target_batch, epochs=1, verbose=0)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

        # Update target network weights every few episodes
        if (episode + 1) % target_update_freq == 0:
            target_model.set_weights(model.get_weights())

# Example usage:
# Make sure that "ver_Monza_2024_laps.csv" is available in your working directory.
filename = "ver_Monza_2024_laps.csv"
env = RaceEnvironment(filename)
episodes = 10
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

train_rsrl_model(env, episodes, gamma, epsilon, epsilon_decay, min_epsilon)
