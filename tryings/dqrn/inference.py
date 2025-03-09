import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Define the binary action space
class RaceAction:
    NO_PIT = 0
    PIT_STOP = 1

    @staticmethod
    def get_action_space():
        return [RaceAction.NO_PIT, RaceAction.PIT_STOP]

# Function to encode the tyre compound (HARD->0, MEDIUM->1, etc.)
def encode_compound(compound):
    return 0 if str(compound).strip().upper() == "HARD" else 1

# Function to convert telemetry dictionary to a state vector
def telemetry_to_state(telemetry):
    # The state vector order must match what was used during training:
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

# Load the saved scaler from scaler.pkl
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("Scaler loaded from scaler.pkl")

# Load the saved model with custom_objects to resolve "mse"
# Note: We pass tf.keras.losses.MeanSquaredError (the class) so that Keras can deserialize the loss.
model = load_model("drqn_race_strategy_model_offline.h5", custom_objects={'mse': tf.keras.losses.MeanSquaredError})
print("Model loaded from drqn_race_strategy_model_offline.h5")

# Function to predict the next race action from telemetry
def predict_next_race_action(telemetry):
    # Convert telemetry dict to state vector
    state = telemetry_to_state(telemetry)
    # Scale the state using the pre-fitted scaler
    state_scaled = scaler.transform(state.reshape(1, -1))
    # Reshape for LSTM input: (batch_size, timesteps, features)
    state_scaled = state_scaled.reshape(1, 1, -1)
    # Predict Q-values using the loaded model
    q_values = model.predict(state_scaled, verbose=0)
    predicted_action = np.argmax(q_values, axis=1)[0]
    return predicted_action, q_values

# Example usage for a single lap's telemetry
if __name__ == '__main__':
    example_telemetry = {
        "LapNumber": 55.0,
        "Sector1Time": 27.5,
        "Sector2Time": 28.7,
        "Sector3Time": 27.3,
        "TyreLife": 5.0,
        "Compound": "HARD",  # or "MEDIUM" depending on the tyre used
        "Position": 8.0,
        "TimeGapToLeader": 3.5,
        "TimeGapToBehind": 0.0,
        "AirTemp": 33.0,
        "TrackTemp": 993.0
    }
    
    action, q_vals = predict_next_race_action(example_telemetry)
    print("Lap 55 predicted action:", "PIT_STOP" if action == RaceAction.PIT_STOP else "NO_PIT")
    print("Q-values:", q_vals)

    # Optionally, if you have a CSV with lap-by-lap telemetry data, you can use:
    """
    def predict_race_telemetry(telemetry_csv):
        data = pd.read_csv(telemetry_csv)
        predictions = []
        for index, row in data.iterrows():
            telemetry = row.to_dict()
            action, q_values = predict_next_race_action(telemetry)
            predictions.append(action)
            print(f"Lap {telemetry['LapNumber']}: Predicted action: {'PIT_STOP' if action == RaceAction.PIT_STOP else 'NO_PIT'}")
        return predictions

    # Replace 'new_race_telemetry.csv' with your CSV file path
    predictions = predict_race_telemetry("new_race_telemetry.csv")
    """
