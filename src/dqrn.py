import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from collections import deque
import random
import os
import glob

def get_csv_files(folder_path):
    return glob.glob(os.path.join(folder_path, "*.csv"))

compound_mapping = {
    "HARD": 0,
    "MEDIUM": 1,
    "SOFT": 2,
}

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
        self.speed_i1 = row["SpeedI1"] if pd.notna(row["SpeedI1"]) else 0.0
        self.speed_i2 = row["SpeedI2"] if pd.notna(row["SpeedI2"]) else 0.0

    def to_array(self):
        return np.array([
            self.lap_number, self.sector1, self.sector2, self.sector3,
            self.tyre_life, self.compound, self.position,
            self.time_gap_leader, self.time_gap_behind,
            self.air_temp, self.track_temp,
            self.speed_i1, self.speed_i2
        ], dtype=np.float32)

# Define a simple binary action space: NO_PIT and PIT_STOP.
class RaceAction:
    NO_PIT = 0
    PIT_STOP = 1

    @staticmethod
    def get_action_space():
        return [RaceAction.NO_PIT, RaceAction.PIT_STOP]

# -----------------------------
# Environment Definitions
# -----------------------------

# Base environment for the race dataset.
# Each row of the CSV represents one lap.
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
        # If a pit stop is taken, add pit penalty to lap time.
        effective_lap_time = lap_time + pit_time if action == RaceAction.PIT_STOP else lap_time
        # Modified reward: higher reward for lower lap times.
        reward = 120 - effective_lap_time
        self.current_index += 1
        done = (self.current_index >= self.total_laps)
        if not done:
            self.state = RaceState(self.data.iloc[self.current_index])
        else:
            self.state = None
        return self.state, reward, done

# DRQN benefits from sequence information.
# This wrapper collects a fixed-length sequence of states.
class RaceEnvironmentSeq:
    def __init__(self, env, seq_length=5):
        self.env = env
        self.seq_length = seq_length
        self.state_seq = deque(maxlen=seq_length)

    def reset(self):
        state = self.env.reset()
        # Fill the sequence with the initial state repeated.
        self.state_seq = deque([state for _ in range(self.seq_length)], maxlen=self.seq_length)
        return self.get_state_seq()

    def step(self, action):
        next_state, reward, done = self.env.step(action)
        self.state_seq.append(next_state)
        return self.get_state_seq(), reward, done

    def get_state_seq(self):
        # Convert the sequence of RaceState objects to a numpy array.
        # If a state is None (episode finished), fill with zeros.
        seq = [
            s.to_array() if s is not None 
            else np.zeros_like(self.state_seq[0].to_array())
            for s in self.state_seq
        ]
        return np.array(seq, dtype=np.float32)

# -----------------------------
# DRQN Model and Training
# -----------------------------

# Build the DRQN model using an LSTM to process the sequence.
def build_drqn_model(seq_length, feature_size, action_space_size):
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_length, feature_size), return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_space_size, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Replay Buffer stores experience tuples: (state_seq, action, reward, next_state_seq, done).
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Training loop for DRQN with sequences, replay buffer, and a target network.
def train_drqn(env_seq, episodes, gamma, epsilon, epsilon_decay, min_epsilon,
               batch_size=32, replay_capacity=1000, target_update_freq=5):
    # Get state shape from an initial sequence.
    initial_seq = env_seq.reset()
    seq_length, feature_size = initial_seq.shape
    action_space = RaceAction.get_action_space()
    action_space_size = len(action_space)

    # Build the main and target networks.
    model = build_drqn_model(seq_length, feature_size, action_space_size)
    target_model = build_drqn_model(seq_length, feature_size, action_space_size)
    target_model.set_weights(model.get_weights())

    replay_buffer = ReplayBuffer(replay_capacity)

    for episode in range(episodes):
        state_seq = env_seq.reset()  # Shape: (seq_length, feature_size)
        total_reward = 0
        done = False

        while not done:
            state_input = state_seq.reshape(1, seq_length, feature_size)
            # Epsilon-greedy action selection.
            if np.random.rand() < epsilon:
                action = np.random.choice(action_space)
            else:
                q_values = model.predict(state_input, verbose=0)
                action = np.argmax(q_values[0])
            next_state_seq, reward, done = env_seq.step(action)
            total_reward += reward

            # Store experience.
            replay_buffer.add((state_seq, action, reward, next_state_seq, done))
            state_seq = next_state_seq

            # Batch update when enough samples are available.
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                state_batch = np.array([exp[0] for exp in batch])
                target_batch = model.predict(state_batch, verbose=0)

                for i, (s_seq, a, r, s_next_seq, d) in enumerate(batch):
                    if d:
                        target_batch[i][a] = r
                    else:
                        next_input = s_next_seq.reshape(1, seq_length, feature_size)
                        t = target_model.predict(next_input, verbose=0)
                        target_batch[i][a] = r + gamma * np.amax(t[0])
                model.fit(state_batch, target_batch, epochs=1, verbose=0)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

        # Update target network weights periodically.
        if (episode+1) % target_update_freq == 0:
            target_model.set_weights(model.get_weights())



if __name__ == "__main__":
    folders = ["Datasets/2021", "Datasets/2022", "Datasets/2023"]
    
    # Initialize the DRQN model outside the loop (so training accumulates)
    # First, load one file to determine state dimensions.
    initial_file = glob.glob(os.path.join(folders[0], "*.csv"))[0]
    base_env = RaceEnvironment(initial_file)
    seq_length = 5
    env_seq = RaceEnvironmentSeq(base_env, seq_length=seq_length)
    initial_seq = env_seq.reset()
    seq_length, feature_size = initial_seq.shape
    action_space_size = len(RaceAction.get_action_space())
    model = build_drqn_model(seq_length, feature_size, action_space_size)
    
    # You may also create a target model and replay buffer if training incrementally.
    
    # Define hyperparameters.
    episodes = 2
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    
    for folder in folders:
        csv_files = get_csv_files(folder)
        for csv_file in csv_files:
            print(f"Training on file: {csv_file}")
            base_env = RaceEnvironment(csv_file)
            env_seq = RaceEnvironmentSeq(base_env, seq_length=seq_length)
            train_drqn(env_seq, episodes, gamma, epsilon, epsilon_decay, min_epsilon)

    
    model.save("drqn_race_strategy_model.h5")
    print("Model saved to drqn_race_strategy_model.h5")
