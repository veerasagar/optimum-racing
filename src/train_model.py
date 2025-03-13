import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

# Define the binary action space.
class RaceAction:
    NO_PIT = 0
    PIT_STOP = 1

    @staticmethod
    def get_action_space():
        return [RaceAction.NO_PIT, RaceAction.PIT_STOP]

# Build the DRQN model.
def build_drqn_model(input_shape, action_space_size):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_space_size, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Load and preprocess the dataset from CSV.
def load_dataset(filename):
    df = pd.read_csv(filename)
    
    # Encode compound: HARD -> 0, others -> 1.
    def encode_compound(compound):
        return 0 if compound.strip().upper() == "HARD" else 1

    states = []
    actions = []
    rewards = []
    lap_numbers = []
    for i in range(len(df) - 1):
        row = df.iloc[i]
        state = np.array([
            row["LapNumber"],
            row["Sector1Time"],
            row["Sector2Time"],
            row["Sector3Time"],
            row["TyreLife"],
            encode_compound(row["Compound"]),
            row["Position"],
            row["TimeGapToLeader"],
            row["TimeGapToBehind"],
            row["AirTemp"],
            row["TrackTemp"]
        ], dtype=np.float32)
        states.append(state)
        lap_numbers.append(row["LapNumber"])
        # Convert PitStop column to action: "true" means pit stop (1), else 0.
        act = 1 if str(row["PitStop"]).strip().lower() == "true" else 0
        actions.append(act)
        pit_time = row["PitTime"] if pd.notna(row["PitTime"]) else 0.0
        effective_lap_time = row["LapTime"] + pit_time
        rewards.append(120 - effective_lap_time)
        
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    # Prepare next_states and terminal flags.
    next_states = states[1:]
    dones = np.zeros(len(states))
    dones[-1] = 1  # Mark the last sample as terminal.
    
    # Align states with next_states.
    states = states[:-1]
    actions = actions[:-1]
    rewards = rewards[:-1]
    dones = dones[:-1]
    lap_numbers = lap_numbers[:-1]

    # Scale states and next_states.
    scaler = StandardScaler()
    states = scaler.fit_transform(states)
    next_states = scaler.transform(next_states)

    return states, actions, rewards, next_states, dones, scaler, lap_numbers, df

# Train the DRQN model using Q-learning update rules.
def train_drqn_model(filename, episodes=10, gamma=0.99, batch_size=32):
    states, actions, rewards, next_states, dones, scaler, lap_numbers, df = load_dataset(filename)
    input_shape = (1, states.shape[1])  # LSTM expects 3D input: (timesteps, features)
    action_space_size = len(RaceAction.get_action_space())
    model = build_drqn_model(input_shape, action_space_size)
    
    # Split the dataset into training and validation sets.
    X_train, X_val, a_train, a_val, r_train, r_val, ns_train, ns_val, d_train, d_val = train_test_split(
        states, actions, rewards, next_states, dones, test_size=0.2, random_state=42
    )
    
    # Reshape for LSTM input.
    X_train = X_train.reshape(-1, 1, states.shape[1])
    X_val = X_val.reshape(-1, 1, states.shape[1])
    ns_train = ns_train.reshape(-1, 1, states.shape[1])
    ns_val = ns_val.reshape(-1, 1, states.shape[1])
    
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        a_train = a_train[indices]
        r_train = r_train[indices]
        ns_train = ns_train[indices]
        d_train = d_train[indices]
        
        for i in range(0, X_train.shape[0], batch_size):
            batch_states = X_train[i:i + batch_size]
            batch_actions = a_train[i:i + batch_size]
            batch_rewards = r_train[i:i + batch_size]
            batch_next_states = ns_train[i:i + batch_size]
            batch_dones = d_train[i:i + batch_size]
            
            current_q_values = model.predict(batch_states, verbose=0)
            next_q_values = model.predict(batch_next_states, verbose=0)
            
            targets = current_q_values.copy()
            for j in range(len(batch_states)):
                act = batch_actions[j]
                if batch_dones[j]:
                    targets[j][act] = batch_rewards[j]
                else:
                    targets[j][act] = batch_rewards[j] + gamma * np.max(next_q_values[j])
            model.fit(batch_states, targets, epochs=1, verbose=0)
        
        val_loss = model.evaluate(X_val, np.zeros((len(X_val), action_space_size)), verbose=0)
        print(f"Validation Loss: {val_loss}")
    
    # Save the trained model and scaler.
    model.save("drqn_race_strategy_model.h5")
    print("Model saved to drqn_race_strategy_model.h5")
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved to scaler.pkl")
    
    return model, scaler, lap_numbers, df

# Visualize results using matplotlib in separate windows and print pit stop statements.
def visualize_results(model, scaler, df):
    # Recreate state vectors from CSV data.
    states = []
    for i in range(len(df) - 1):
        row = df.iloc[i]
        compound_encoded = 0 if row["Compound"].strip().upper() == "HARD" else 1
        state = np.array([
            row["LapNumber"],
            row["Sector1Time"],
            row["Sector2Time"],
            row["Sector3Time"],
            row["TyreLife"],
            compound_encoded,
            row["Position"],
            row["TimeGapToLeader"],
            row["TimeGapToBehind"],
            row["AirTemp"],
            row["TrackTemp"]
        ], dtype=np.float32)
        states.append(state)
    states = np.array(states)
    states_scaled = scaler.transform(states)
    states_reshaped = states_scaled.reshape(-1, 1, states_scaled.shape[1])
    
    # Predict Q-values and derive pit stop actions.
    q_values = model.predict(states_reshaped, verbose=0)
    predicted_actions = np.argmax(q_values, axis=1)
    
    # Create a DataFrame for visualization (ensuring alignment with predictions).
    df_vis = df.iloc[:len(predicted_actions)].copy()
    df_vis["PredictedAction"] = predicted_actions
    # Convert the actual PitStop column to a numeric flag.
    df_vis["ActualPitStop"] = df_vis["PitStop"].apply(lambda x: 1 if str(x).strip().lower() == "true" else 0)
    
    # --- Print Actual Pit Stop Statements ---
    actual_pit = df_vis[df_vis["ActualPitStop"] == 1]
    print("\nActual Pit Stops:")
    for idx, row in actual_pit.iterrows():
        print(f"Lap {row['LapNumber']}: Lap Time = {row['LapTime']} sec, Compound = {row['Compound']}")
    
    # --- Print Predicted Pit Stop Statements ---
    predicted_pit = df_vis[df_vis["PredictedAction"] == RaceAction.PIT_STOP]
    print("\nPredicted Pit Stops:")
    for idx, row in predicted_pit.iterrows():
        print(f"Lap {row['LapNumber']}: Lap Time = {row['LapTime']} sec, Compound = {row['Compound']}")
    
    # --- Figure 1: Actual Race Lap Times with Actual Pit Stops ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_vis["LapNumber"], df_vis["LapTime"], label="Lap Time", color="blue", linestyle="-", marker="o")
    plt.scatter(actual_pit["LapNumber"], actual_pit["LapTime"], color="red", s=100, label="Actual Pit Stop")
    plt.title("Actual Race Lap Times with Actual Pit Stops")
    plt.xlabel("Lap Number")
    plt.ylabel("Lap Time (seconds)")
    plt.legend()
    plt.grid(True)
    
    # --- Figure 2: Race Lap Times with Predicted Pit Stops ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_vis["LapNumber"], df_vis["LapTime"], label="Lap Time", color="blue", linestyle="-", marker="o")
    plt.scatter(predicted_pit["LapNumber"], predicted_pit["LapTime"], color="green", s=100, label="Predicted Pit Stop")
    plt.title("Race Lap Times with Predicted Pit Stops")
    plt.xlabel("Lap Number")
    plt.ylabel("Lap Time (seconds)")
    plt.legend()
    plt.grid(True)
    
    # Display the graphs in separate windows.
    plt.show()

# Main execution.
if __name__ == '__main__':
    filename = "ver_Monza_2024_laps.csv"  # Replace with your CSV file path.
    episodes = 10
    gamma = 0.99
    batch_size = 32

    model, scaler, lap_numbers, df = train_drqn_model(filename, episodes, gamma, batch_size)
    visualize_results(model, scaler, df)
