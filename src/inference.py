import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Define the binary action space.
class RaceAction:
    NO_PIT = 0
    PIT_STOP = 1

    @staticmethod
    def get_action_space():
        return [RaceAction.NO_PIT, RaceAction.PIT_STOP]

# Function to encode the tyre compound (HARD -> 0, others -> 1).
def encode_compound(compound):
    return 0 if str(compound).strip().upper() == "HARD" else 1

# Function to convert a telemetry dictionary to a state vector.
def telemetry_to_state(telemetry):
    # Ensure the state vector order matches what was used during training:
    # [LapNumber, Sector1Time, Sector2Time, Sector3Time, TyreLife, Compound,
    #  Position, TimeGapToLeader, TimeGapToBehind, AirTemp, TrackTemp]
    state = np.array([
        telemetry["LapNumber"],
        telemetry["Sector1Time"],
        telemetry["Sector2Time"],
        telemetry["Sector3Time"],
        telemetry["TyreLife"],
        encode_compound(telemetry["Compound"]),
        telemetry["Position"],
        telemetry["TimeGapToLeader"],
        telemetry["TimeGapToBehind"],
        telemetry["AirTemp"],
        telemetry["TrackTemp"]
    ], dtype=np.float32)
    return state

# Load the pre-fitted scaler.
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("Scaler loaded from scaler.pkl")

# Load the saved DRQN model (ensure the correct model file is used).
model = load_model("drqn_race_strategy_model_offline.h5", custom_objects={'mse': tf.keras.losses.MeanSquaredError})
print("Model loaded from drqn_race_strategy_model_offline.h5")

# Function to predict the next race action given telemetry.
def predict_next_race_action(telemetry):
    state = telemetry_to_state(telemetry)
    # Check for missing values in state
    if np.isnan(state).any():
        raise ValueError("Telemetry state contains NaN values. Check the telemetry data.")
    # Print state before scaling for debugging (optional)
    # print("State before scaling:", state)
    state_scaled = scaler.transform(state.reshape(1, -1))
    # Print state after scaling for debugging (optional)
    # print("State after scaling:", state_scaled)
    # Reshape to (batch_size, timesteps, features) as expected by the LSTM.
    state_scaled = state_scaled.reshape(1, 1, -1)
    q_values = model.predict(state_scaled, verbose=0)
    predicted_action = np.argmax(q_values, axis=1)[0]
    return predicted_action, q_values

if __name__ == '__main__':
    # Load the dataset.
    dataset_file = "ver_Monza_2024_laps.csv"  # Adjust path if necessary.
    df = pd.read_csv(dataset_file)
    
    # Define required telemetry columns.
    required_columns = [
        "LapNumber", "Sector1Time", "Sector2Time", "Sector3Time",
        "TyreLife", "Compound", "Position", "TimeGapToLeader",
        "TimeGapToBehind", "AirTemp", "TrackTemp"
    ]
    
    # Drop rows with missing required fields.
    df_clean = df.dropna(subset=required_columns)
    if df_clean.empty:
        print("No valid telemetry data available in the dataset.")
    else:
        # Pick a random valid lap.
        random_row = df_clean.sample(n=1).iloc[0].to_dict()
        print("Randomly selected telemetry data (Lap {}):".format(random_row["LapNumber"]))
        print(random_row)
        
        # Predict the action.
        action, q_vals = predict_next_race_action(random_row)
        action_str = "PIT_STOP" if action == RaceAction.PIT_STOP else "NO_PIT"
        print("\nPredicted action for Lap {}: {}".format(random_row["LapNumber"], action_str))
        print("Q-values:", q_vals)
