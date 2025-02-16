import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from dqrn import RaceEnvironment, RaceAction  # Use your existing classes
# Define the DRQN model
def build_drqn_model(input_shape, action_space_size):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))  # No return_sequences
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_space_size, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Load and preprocess the dataset
def load_dataset(filename):
    df = pd.read_csv(filename)
    
    # Extract state, action, reward, next_state, and done columns
    state_columns = [
        "track", "safety_car", "position", "tyre", "tyre_degradation",
        "gap1", "gap2", "gap3", "last_lap_time"
    ]
    next_state_columns = [
        "next_track", "next_safety_car", "next_position", "next_tyre",
        "next_tyre_degradation", "next_gap1", "next_gap2", "next_gap3",
        "next_last_lap_time"
    ]
    
    # Convert to numpy arrays
    states = df[state_columns].values
    actions = df["action"].values
    rewards = df["reward"].values
    next_states = df[next_state_columns].values
    dones = df["done"].values
    
    return states, actions, rewards, next_states, dones

# Train the DRQN model
def train_drqn_model(filename, episodes=100, gamma=0.99, batch_size=32):
    # Load the dataset
    states, actions, rewards, next_states, dones = load_dataset(filename)
    
    # Define input shape and action space size
    input_shape = (1, states.shape[1])  # (timesteps, features)
    action_space_size = len(RaceAction.get_action_space())
    
    # Build the DRQN model
    model = build_drqn_model(input_shape, action_space_size)
    
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(states, actions, test_size=0.2, random_state=42)
    
    # Reshape data for LSTM input (batch_size, timesteps, features)
    X_train = X_train.reshape(-1, 1, states.shape[1])
    X_val = X_val.reshape(-1, 1, states.shape[1])
    next_states = next_states.reshape(-1, 1, states.shape[1])
    
    # Training loop
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        
        # Shuffle the training data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        rewards = rewards[indices]
        next_states = next_states[indices]
        dones = dones[indices]
        
        # Train in mini-batches
        for i in range(0, X_train.shape[0], batch_size):
            batch_states = X_train[i:i + batch_size]
            batch_actions = y_train[i:i + batch_size]
            batch_rewards = rewards[i:i + batch_size]
            batch_next_states = next_states[i:i + batch_size]
            batch_dones = dones[i:i + batch_size]
            
            # Predict Q-values for current and next states
            current_q_values = model.predict(batch_states)
            next_q_values = model.predict(batch_next_states)
            
            # Update Q-values using the Bellman equation
            targets = current_q_values.copy()
            for j in range(len(batch_states)):
                action = batch_actions[j]
                if batch_dones[j]:
                    targets[j][action] = batch_rewards[j]
                else:
                    targets[j][action] = batch_rewards[j] + gamma * np.max(next_q_values[j])
            
            # Train the model on the updated Q-values
            model.fit(batch_states, targets, verbose=0)
        
        # Validate the model (example placeholder)
        val_loss = model.evaluate(X_val, np.zeros((len(X_val), action_space_size)), verbose=0)
        print(f"Validation Loss: {val_loss}")
    
    # Save the trained model
    model.save("drqn_race_strategy_model.h5")
    print("Model saved to drqn_race_strategy_model.h5")

# Parameters
filename = "race_strategy_data.csv"
episodes = 10
gamma = 0.99
batch_size = 32

# Train the model
train_drqn_model(filename, episodes, gamma, batch_size)