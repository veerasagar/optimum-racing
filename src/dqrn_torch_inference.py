import numpy as np
import pandas as pd
import torch
import os
import glob
from matplotlib import pyplot as plt
import fastf1
import fastf1.plotting

def map_action_to_compound(action):
    """Map action index to compound string"""
    compound_map = {
        1: "HARD",
        2: "MEDIUM",
        3: "SOFT",
        4: "INTERMEDIATE",
        5: "WET"
    }
    return compound_map.get(action, "NO_PIT")

def simulate_race(model, env_seq, device="cpu"):
    """
    Modified version with mandatory pit stop enforcement
    """
    model.eval()
    state_seq = env_seq.reset()
    total_reward = 0
    pit_stops = []
    position_history = []
    done = False
    forced_pit = False
    total_laps = env_seq.env.total_laps

    while not done:
        current_lap = env_seq.env.current_index + 1  # Laps are 0-indexed
        
        # Force pit stop if not done by lap 50 (or 80% of race distance)
        if not pit_stops and current_lap >= int(total_laps * 0.8) and not forced_pit:
            print("\nMandatory pit stop enforced by race rules!")
            action = RaceAction.PIT_MEDIUM  # Choose safest available compound
            forced_pit = True
        else:
            state_tensor = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values, dim=1).item()

        # Get current state information
        current_state = env_seq.env.state
        lap_number = int(current_state.lap_number)
        current_position = int(current_state.position)
        position_history.append((lap_number, current_position))

        # Handle pit action
        if action != RaceAction.NO_PIT or forced_pit:
            compound = map_action_to_compound(action)
            remaining = current_state.remaining_compounds[compound_mapping[compound]]
            
            if remaining > 0 or forced_pit:
                pit_stops.append((lap_number, compound, remaining))
                print(f"Lap {lap_number}: Pit stop with {compound} (Remaining: {remaining})")
                forced_pit = False  # Reset forced pit flag
            else:
                print(f"Lap {lap_number}: Invalid pit attempt - no {compound} tires remaining")
                action = RaceAction.NO_PIT  # Override invalid action

        # Execute action
        state_seq, reward, done = env_seq.step(action)
        total_reward += reward

        # Apply penalty if no pit stops in last 3 laps
        if done and not pit_stops:
            print("\nRace regulations violation! No pit stops performed.")
            total_reward -= 50  # Heavy penalty
            # Force a pit stop in the last lap
            last_lap = total_laps - 1
            pit_stops.append((last_lap, "MEDIUM", 0))
            print("Emergency pit stop recorded for race compliance")

    return pit_stops, position_history, total_reward

def print_race_report(pit_stops, position_history):
    """Print formatted race simulation results"""
    print("\nRace Strategy Report:")
    print("=====================")
    
    print("\nPit Stop Summary:")
    if not pit_stops:
        print("No pit stops performed")
    else:
        for stop in pit_stops:
            print(f"Lap {stop[0]}: {stop[1]} compound (Remaining sets: {stop[2]})")

    print("\nPosition Changes:")
    prev_pos = position_history[0][1] if position_history else 0
    for lap, pos in position_history:
        change = prev_pos - pos
        if change > 0:
            print(f"Lap {lap}: Gained {change} positions (Current: {pos})")
        elif change < 0:
            print(f"Lap {lap}: Lost {abs(change)} positions (Current: {pos})")
        else:
            print(f"Lap {lap}: Maintained position ({pos})")
        prev_pos = pos

fastf1.plotting.setup_mpl()

# List of races to analyze

# Main simulation execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and environment (same as before)
    model = DRQN(seq_length=5, feature_size=18, action_space_size=6).to(device)
    model.load_state_dict(torch.load("drqn_race_strategy_model.pth", map_location=device))
    
    test_files = glob.glob(os.path.join("../Datasets/2024", "*.csv"))
    for race_file in test_files:
        base_env = RaceEnvironment(race_file)
        env_seq = RaceEnvironmentSeq(base_env, seq_length=5)
        
        pit_stops, positions, reward = simulate_race(model, env_seq, device)
        print_race_report(pit_stops, positions)

    races = ["Netherlands", "Italy", "Azerbaijan", "Singapore", "United States",
         "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"]

# Dictionary to hold Verstappen's stint data for each race and year
    verstappen_data = {}

    for race in races:
        verstappen_data[race] = {}
        for year in [2023, 2024]:
            try:
                # Load the race session
                session = fastf1.get_session(year, race, 'R')
                session.load()
                
                # Get all laps and process stints
                laps = session.laps
                stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
                stints = stints.groupby(["Driver", "Stint", "Compound"]).agg(
                    StintLength=('LapNumber', 'count'),
                    StartLap=('LapNumber', 'min'),
                    EndLap=('LapNumber', 'max')
                ).reset_index()
                
                # Filter for Verstappen's stints
                verstappen_stints = stints[stints["Driver"] == "VER"]
                verstappen_data[race][year] = verstappen_stints
                
            except Exception as e:
                print(f"Error loading {year} {race}: {e}")
                verstappen_data[race][year] = None

    # Plotting
    for race in races:
        data_2024 = verstappen_data[race].get(2024)
        data_2023 = verstappen_data[race].get(2023)
        
        # Skip if data is missing or empty
        if data_2024 is None or data_2023 is None:
            print(f"Skipping {race} due to missing data")
            continue
        if data_2024.empty or data_2023.empty:
            print(f"Skipping {race} due to empty data in one year")
            continue
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle(f"Verstappen Tyre Strategy Comparison - {race} GP", y=1.05)
        
        # Plot 2024 stints
        for _, stint in data_2024.iterrows():
            color = fastf1.plotting.COMPOUND_COLORS.get(stint['Compound'], 'grey')
            start = stint['StartLap'] - 0.5
            ax1.barh(0, stint['StintLength'], left=start, color=color, height=0.5, edgecolor='black')
        
        # Plot 2023 stints
        for _, stint in data_2023.iterrows():
            color = fastf1.plotting.COMPOUND_COLORS.get(stint['Compound'], 'grey')
            start = stint['StartLap'] - 0.5
            ax2.barh(0, stint['StintLength'], left=start, color=color, height=0.5, edgecolor='black')
        
        # Configure axes
        ax1.set_title("Actual", pad=10)
        ax2.set_title("Predicted", pad=10)
        ax1.set_yticks([])
        ax2.set_yticks([])
        ax2.set_xlabel("Lap Number")
        
        # Set x-axis limits
        max_lap = max(data_2024['EndLap'].max(), data_2023['EndLap'].max())
        ax1.set_xlim(0, max_lap + 1)
        
        # Create legend
        compounds = set(data_2024['Compound']).union(set(data_2023['Compound']))
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, color=fastf1.plotting.COMPOUND_COLORS[c]) 
            for c in compounds if c in fastf1.plotting.COMPOUND_COLORS
        ]
        ax1.legend(legend_elements, compounds, title="Tyre Compounds", loc="upper right")
        
        plt.tight_layout()
        plt.show()