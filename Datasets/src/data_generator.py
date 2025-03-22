import fastf1 as ff1
import pandas as pd
import numpy as np
from datetime import timedelta

def get_track_status(lap_time, status_df):
    # Finds the track status corresponding to a lap time by checking where
    # lap_time falls between two successive status change times.
    mask = ((status_df['Time'] <= lap_time) &
            (lap_time < status_df['Time'].shift(-1).fillna(pd.Timedelta.max)))
    relevant_status = status_df[mask]
    if not relevant_status.empty:
        return relevant_status.iloc[-1]['Status']
    return np.nan

def process_session(year, event, session_type, drivers=None):
    """
    Process a FastF1 session for the given year, event (circuit), and session type.
    Optionally, a driver or list of drivers (using their driver numbers as strings)
    can be provided. If not provided, all drivers in the session will be processed.
    The data for each driver is saved as a CSV file with a name including the driver's
    name, event, and year.
    """
    # Load the session
    session = ff1.get_session(year, event, session_type)
    session.load()

    all_laps = session.laps
    weather = session.weather_data
    track_status_df = session.track_status

    # Define session start as the earliest lap time in the session
    session_start_time = all_laps['Time'].min()

    # If no drivers are provided, use all drivers from the session.
    if drivers is None:
        drivers = list(set(all_laps['DriverNumber'].astype(str)))
    else:
        # If a single driver is passed as a string, convert it to a list.
        if isinstance(drivers, str):
            drivers = [drivers]

    # Pre-calculate cumulative lap times for every driver (needed for gap calculations)
    unique_drivers = all_laps['DriverNumber'].unique()
    cumulative_times = {str(driver): [] for driver in unique_drivers}
    for driver in unique_drivers:
        driver_laps = all_laps[all_laps['DriverNumber'] == driver].sort_values(by='LapNumber')
        cumulative = timedelta(0)
        for lap in driver_laps.itertuples():
            cumulative += lap.LapTime
            cumulative_times[str(driver)].append((lap.LapNumber, cumulative))

    # Process each requested driver
    for driver in drivers:
        # Select laps for the current driver
        driver_laps = all_laps[all_laps['DriverNumber'] == driver]
        if driver_laps.empty:
            print(f"No laps found for driver {driver}. Skipping.")
            continue

        # Merge with weather data
        driver_laps = pd.merge_asof(driver_laps.sort_values("Time"),
                                    weather.sort_values("Time"),
                                    on="Time")

        # Get the driver name from the first lap data (e.g., driver's abbreviation)
        driver_name = driver_laps.iloc[0]['Driver'].replace(" ", "_").lower()

        data = []
        for idx in range(len(driver_laps)):
            lap = driver_laps.iloc[idx]
            lap_number = lap.LapNumber
            lap_time = lap.LapTime.total_seconds()
            compound = lap.Compound
            tyre_life = lap.TyreLife
            position = lap.Position

            # Calculate lap time relative to session start
            lap_relative_time = lap.Time - session_start_time
            track_status = get_track_status(lap_relative_time, track_status_df)

            sector1 = lap.Sector1Time.total_seconds()
            sector2 = lap.Sector2Time.total_seconds()
            sector3 = lap.Sector3Time.total_seconds()

            pit_stop = not pd.isnull(lap.PitInTime)
            if pit_stop and track_status == '1' and (idx + 1) < len(driver_laps):
                pit_time = (driver_laps.iloc[idx+1].PitOutTime - lap.PitInTime).total_seconds()
            else:
                pit_time = 0

            speed_i1 = lap.SpeedI1
            speed_i2 = lap.SpeedI2
            speed_fl = lap.SpeedFL
            speed_st = lap.SpeedST

            personal_best = driver_laps['LapTime'][:idx+1].min().total_seconds()

            # Find cumulative time for this driver on the current lap
            ver_cumulative = next((ct for ln, ct in cumulative_times[str(driver)] if ln == lap_number), None)
            # Determine the fastest (leader's) cumulative time for this lap among all drivers
            leader_time = min([ct for times in cumulative_times.values()
                               for (ln, ct) in times if ln == lap_number], default=None)
            gap_leader = (ver_cumulative - leader_time).total_seconds() if leader_time is not None else 0

            # Determine gap to the driver immediately behind (if available)
            behind_drivers = []
            for d in cumulative_times:
                if d == str(driver):
                    continue
                ct_value = next((ct for ln, ct in cumulative_times[d] if ln == lap_number), None)
                if ct_value is not None:
                    other_laps = all_laps[(all_laps['DriverNumber'] == int(d)) & (all_laps['LapNumber'] == lap_number)]
                    if not other_laps.empty:
                        other_position = other_laps.iloc[0]['Position']
                        if other_position == position + 1:
                            behind_drivers.append(ct_value)
            gap_behind = (min(behind_drivers) - ver_cumulative).total_seconds() if behind_drivers else 0

            air_temp = lap.AirTemp
            humidity = lap.Humidity
            pressure = lap.Pressure
            track_temp = lap.TrackTemp
            wind_direction = lap.WindDirection
            wind_speed = lap.WindSpeed

            data.append({
                'LapNumber': lap_number,
                'LapTime': lap_time,
                'PitStop': pit_stop,
                'PitTime': pit_time,
                'Sector1Time': sector1,
                'Sector2Time': sector2,
                'Sector3Time': sector3,
                'SpeedI1': speed_i1,
                'SpeedI2': speed_i2,
                'SpeedFL': speed_fl,
                'SpeedST': speed_st,
                'PersonalBestTime': personal_best,
                'Compound': compound,
                'TyreLife': tyre_life,
                'TrackStatus': track_status,
                'Position': position,
                'TimeGapToLeader': gap_leader,
                'TimeGapToBehind': gap_behind,
                'AirTemp': air_temp,
                'Humidity': humidity,
                'Pressure': pressure,
                'TrackTemp': track_temp,
                'WindDirection': wind_direction,
                'WindSpeed': wind_speed
            })

        df = pd.DataFrame(data)
        file_name = f"./datasets/{year}/{driver_name}_{event}_{year}_laps.csv"
        df.to_csv(file_name, index=False)
        print(f"Data saved to {file_name}")

# Example usage:
if __name__ == "__main__":
    # Define your parameters: year, session type, and driver(s) by their number (as strings). For multiple drivers, pass a list.
    year = 2021
    session_type = 'R'
    drivers = ['33']  # Replace with the driver numbers you want to process

    # Get the event schedule for the specified year
    event_schedule = ff1.get_event_schedule(year)

    # Iterate over each event in the schedule
    for event_name in event_schedule['EventName']:
        print(f"Processing event: {event_name}")
        process_session(year, event_name, session_type, drivers)
