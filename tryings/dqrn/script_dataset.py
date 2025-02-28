import numpy as np
import pandas as pd
from dqrn import RaceEnvironment, RaceAction  # Use your existing classes

def generate_dataset(num_episodes=100, filename="race_strategy_data.csv"):
    env = RaceEnvironment()
    data = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = np.random.choice(RaceAction.get_action_space())
            next_state, reward, done = env.step(action)
            
            # Extract current state
            current_state = state.to_array()
            
            # Extract next state
            next_state_array = next_state.to_array()
            
            # Append transition to dataset
            data.append([
                *current_state.tolist(),  # Track, SafetyCar, Position, Tyre, TyreDegradation, Gaps (3), LastLapTime
                action,
                reward,
                *next_state_array.tolist(),  # Next state
                int(done)
            ])
            
            state = next_state

    # Define column names
    columns = [
        "track", "safety_car", "position", "tyre", "tyre_degradation",
        "gap1", "gap2", "gap3", "last_lap_time",
        "action", "reward",
        "next_track", "next_safety_car", "next_position", "next_tyre",
        "next_tyre_degradation", "next_gap1", "next_gap2", "next_gap3",
        "next_last_lap_time", "done"
    ]
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")

# Generate 100 episodes of synthetic data
generate_dataset(num_episodes=100)