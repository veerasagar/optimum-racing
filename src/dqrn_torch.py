import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import glob

# -----------------------------
# Data and Environment Classes
# -----------------------------

def get_csv_files(folder_path):
    return glob.glob(os.path.join(folder_path, "*.csv"))

compound_mapping = {
    "HARD": 0,
    "MEDIUM": 1,
    "SOFT": 2,
    "INTERMEDIATE": 3,
    "WET": 4
}

degradation_rates = [0.1, 0.2, 0.3, 0.15, 0.1]  # per lap time increase per tire life

class RaceState:
    def __init__(self, row, current_compound, remaining_compounds, tire_life):
        self.lap_number = row["LapNumber"] if pd.notna(row["LapNumber"]) else 0.0
        self.sector1 = row["Sector1Time"] if pd.notna(row["Sector1Time"]) else 0.0
        self.sector2 = row["Sector2Time"] if pd.notna(row["Sector2Time"]) else 0.0
        self.sector3 = row["Sector3Time"] if pd.notna(row["Sector3Time"]) else 0.0
        self.tyre_life = tire_life
        self.compound = current_compound
        self.remaining_compounds = remaining_compounds.copy()
        self.position = row["Position"] if pd.notna(row["Position"]) else 0.0
        self.time_gap_leader = row["TimeGapToLeader"] if pd.notna(row["TimeGapToLeader"]) else 0.0
        self.time_gap_behind = row["TimeGapToBehind"] if pd.notna(row["TimeGapToBehind"]) else 0.0
        self.air_temp = row["AirTemp"] if pd.notna(row["AirTemp"]) else 0.0
        self.track_temp = row["TrackTemp"] if pd.notna(row["TrackTemp"]) else 0.0
        self.speed_i1 = row["SpeedI1"] if pd.notna(row["SpeedI1"]) else 0.0
        self.speed_i2 = row["SpeedI2"] if pd.notna(row["SpeedI2"]) else 0.0

    def to_array(self):
        return np.concatenate([
            np.array([
                self.lap_number,
                self.sector1,
                self.sector2,
                self.sector3,
                self.tyre_life,
                self.compound,
                self.position,
                self.time_gap_leader,
                self.time_gap_behind,
                self.air_temp,
                self.track_temp,
                self.speed_i1,
                self.speed_i2
            ], dtype=np.float32),
            np.array(self.remaining_compounds, dtype=np.float32)
        ])

class RaceAction:
    NO_PIT = 0
    PIT_HARD = 1
    PIT_MEDIUM = 2
    PIT_SOFT = 3
    PIT_INTERMEDIATE = 4
    PIT_WET = 5

    @staticmethod
    def get_action_space():
        return [RaceAction.NO_PIT, RaceAction.PIT_HARD, RaceAction.PIT_MEDIUM,
                RaceAction.PIT_SOFT, RaceAction.PIT_INTERMEDIATE, RaceAction.PIT_WET]

class RaceEnvironment:
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.total_laps = len(self.data)
        self.current_index = 0
        self.state = None
        self.has_pitted = False
        self.tire_life = 0
        self.current_compound = None
        self.remaining_compounds = [3] * len(compound_mapping)
        self.previous_position = None
        
        # Initialize starting compound
        starting_compound_str = self.data.iloc[0]["Compound"].strip().upper() if pd.notna(self.data.iloc[0]["Compound"]) else "HARD"
        self.current_compound = compound_mapping.get(starting_compound_str, 0)
        self.remaining_compounds[self.current_compound] -= 1

    def reset(self):
        self.current_index = 0
        self.has_pitted = False
        self.tire_life = 0
        starting_compound_str = self.data.iloc[0]["Compound"].strip().upper() if pd.notna(self.data.iloc[0]["Compound"]) else "HARD"
        self.current_compound = compound_mapping.get(starting_compound_str, 0)
        self.remaining_compounds = [3] * len(compound_mapping)
        self.remaining_compounds[self.current_compound] -= 1
        self.state = self._create_race_state(self.data.iloc[self.current_index])
        self.previous_position = self.state.position
        return self.state

    def _create_race_state(self, row):
        return RaceState(row, self.current_compound, self.remaining_compounds, self.tire_life)

    def step(self, action):
        row = self.data.iloc[self.current_index]
        base_lap_time = row["LapTime"] if pd.notna(row["LapTime"]) else 120.0
        pit_time = row["PitTime"] if pd.notna(row["PitTime"]) else 20.0
        degradation = degradation_rates[self.current_compound] * self.tire_life
        effective_lap_time = base_lap_time + degradation
        reward = 120 - effective_lap_time
        compound_changed = False

        if action != RaceAction.NO_PIT:
            compound_idx = action - 1
            if self.remaining_compounds[compound_idx] > 0:
                effective_lap_time += pit_time
                self.current_compound = compound_idx
                self.remaining_compounds[compound_idx] -= 1
                self.tire_life = 0
                self.has_pitted = True
                compound_changed = True
            else:
                reward -= 10  # Penalize invalid pit

        if not compound_changed:
            self.tire_life += 1

        # Update position-based reward
        current_position = self.state.position
        position_change = self.previous_position - current_position
        reward += position_change * 2.0  # Reward for position gain
        self.previous_position = current_position

        self.current_index += 1
        done = self.current_index >= self.total_laps

        if not done:
            self.state = self._create_race_state(self.data.iloc[self.current_index])
        else:
            self.state = None

        return self.state, reward, done

class RaceEnvironmentSeq:
    def __init__(self, env, seq_length=5):
        self.env = env
        self.seq_length = seq_length
        self.state_seq = deque(maxlen=seq_length)

    def reset(self):
        state = self.env.reset()
        self.state_seq = deque([state for _ in range(self.seq_length)], maxlen=self.seq_length)
        return self.get_state_seq()

    def step(self, action):
        next_state, reward, done = self.env.step(action)
        self.state_seq.append(next_state)
        return self.get_state_seq(), reward, done

    def get_state_seq(self):
        seq = [
            s.to_array() if s is not None
            else np.zeros_like(self.state_seq[0].to_array())
            for s in self.state_seq
        ]
        return np.array(seq, dtype=np.float32)

# -----------------------------
# DRQN Model
# -----------------------------

class DRQN(nn.Module):
    def __init__(self, seq_length, feature_size, action_space_size):
        super(DRQN, self).__init__()
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.hidden_size = 64
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=self.hidden_size, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, action_space_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# Training Loop
# -----------------------------

def train_drqn(env_seq, episodes, gamma, epsilon, epsilon_decay, min_epsilon,
               batch_size=32, replay_capacity=1000, target_update_freq=5, device="cpu"):
    initial_seq = env_seq.reset()
    seq_length, feature_size = initial_seq.shape
    action_space = RaceAction.get_action_space()
    action_space_size = len(action_space)

    model = DRQN(seq_length, feature_size, action_space_size).to(device)
    target_model = DRQN(seq_length, feature_size, action_space_size).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(replay_capacity)

    for episode in range(episodes):
        state_seq = env_seq.reset()
        total_reward = 0
        done = False

        while not done:
            state_input = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(device)
            if np.random.rand() < epsilon:
                action = random.choice(action_space)
            else:
                with torch.no_grad():
                    q_values = model(state_input)
                    action = torch.argmax(q_values, dim=1).item()
            next_state_seq, reward, done = env_seq.step(action)
            total_reward += reward

            replay_buffer.add((state_seq, action, reward, next_state_seq, done))
            state_seq = next_state_seq

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                state_batch = torch.tensor(np.array([exp[0] for exp in batch]), dtype=torch.float32).to(device)
                actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long).to(device)
                rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32).to(device)
                next_state_batch = torch.tensor(np.array([exp[3] for exp in batch]), dtype=torch.float32).to(device)
                dones = torch.tensor([exp[4] for exp in batch], dtype=torch.float32).to(device)

                q_values = model(state_batch).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_model(next_state_batch)
                    max_next_q_values, _ = torch.max(next_q_values, dim=1)
                    target = rewards + gamma * max_next_q_values * (1 - dones)

                loss = loss_fn(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

        if (episode+1) % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

    return model




# -----------------------------
# Main Execution
# -----------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folders = ["../Datasets/2021", "../Datasets/2022", "../Datasets/2023"]

    initial_file = glob.glob(os.path.join(folders[0], "*.csv"))[0]
    base_env = RaceEnvironment(initial_file)
    seq_length = 5
    env_seq = RaceEnvironmentSeq(base_env, seq_length=seq_length)
    initial_seq = env_seq.reset()
    seq_length, feature_size = initial_seq.shape
    action_space_size = len(RaceAction.get_action_space())

    episodes = 10
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01

    shared_model = DRQN(seq_length, feature_size, action_space_size).to(device)

    for folder in folders:
        csv_files = get_csv_files(folder)
        for csv_file in csv_files:
            print(f"Training on file: {csv_file}")
            base_env = RaceEnvironment(csv_file)
            env_seq = RaceEnvironmentSeq(base_env, seq_length=seq_length)
            shared_model = train_drqn(env_seq, episodes, gamma, epsilon, epsilon_decay, min_epsilon, device=device)

    torch.save(shared_model.state_dict(), "drqn_race_strategy_model.pth")
    print("Model saved to drqn_race_strategy_model.pth")