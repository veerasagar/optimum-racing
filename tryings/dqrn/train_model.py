import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Define the binary action space
class RaceAction:
    NO_PIT = 0
    PIT_STOP = 1

    @staticmethod
    def get_action_space():
        return [RaceAction.NO_PIT, RaceAction.PIT_STOP]

# Build the DRQN model
def build_drqn_model(input_shape, action_space_size):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_space_size, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Load and preprocess the dataset from CSV
def load_dataset(filename):
    df = pd.read_csv(filename)
    
    def encode_compound(compound):
        return 0 if compound.strip().upper() == "HARD" else 1

    states = []
    actions = []
    rewards = []
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
        ])
        states.append(state)
        act = 1 if str(row["PitStop"]).strip().lower() == "true" else 0
        actions.append(act)
        rewards.append(-row["LapTime"])
        
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    next_states = states[1:]
    dones = np.zeros(len(states))
    dones[-1] = 1

    states = states[:-1]
    actions = actions[:-1]
    rewards = rewards[:-1]
    dones = dones[:-1]
    next_states = next_states

    return states, actions, rewards, next_states, dones

# Train and evaluate the DRQN model
def train_drqn_model(filename, episodes=10, gamma=0.99, batch_size=32):
    states, actions, rewards, next_states, dones = load_dataset(filename)
    
    input_shape = (1, states.shape[1])
    action_space_size = len(RaceAction.get_action_space())
    model = build_drqn_model(input_shape, action_space_size)
    
    # Split dataset
    X_train, X_val, a_train, a_val, r_train, r_val, ns_train, ns_val, d_train, d_val = train_test_split(
        states, actions, rewards, next_states, dones, test_size=0.2, random_state=42
    )
    
    # Reshape for LSTM
    X_train = X_train.reshape(-1, 1, states.shape[1])
    X_val = X_val.reshape(-1, 1, states.shape[1])
    ns_train = ns_train.reshape(-1, 1, states.shape[1])
    ns_val = ns_val.reshape(-1, 1, states.shape[1])
    
    # Training loop
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

    # Final evaluation
    print("\nFinal Evaluation on Validation Set:")
    val_q_values = model.predict(X_val, verbose=0)
    predicted_actions = np.argmax(val_q_values, axis=1)
    true_actions = a_val
    
    accuracy = accuracy_score(true_actions, predicted_actions)
    f1 = f1_score(true_actions, predicted_actions)
    precision = precision_score(true_actions, predicted_actions)
    recall = recall_score(true_actions, predicted_actions)
    conf_matrix = confusion_matrix(true_actions, predicted_actions)
    
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    print("\nClassification Report:")
    print(classification_report(true_actions, predicted_actions, 
                              target_names=['NO_PIT', 'PIT_STOP']))

    model.save("drqn_race_strategy_model.h5")
    print("\nModel saved to drqn_race_strategy_model.h5")

# Example usage
filename = "ver_Monza_2024_laps.csv"  # Replace with your CSV path
episodes = 10
gamma = 0.99
batch_size = 32

train_drqn_model(filename, episodes, gamma, batch_size)