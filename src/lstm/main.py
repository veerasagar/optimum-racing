import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import dash
from dash import html, dcc, Input, Output
import plotly.express as px

# Step 1: Data Collection
def fetch_data():
    """
    Loads Formula 1 lap data from a CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing lap data and corresponding weather
                          information, or None if data fetching fails.
    """
    try:
        # Load data from CSV file
        df = pd.read_csv('sample_f1_data.csv')
        print("Data loaded successfully from CSV")
        return df
    except Exception as e:
        print(f"Failed to load data from CSV: {e}")
        return None

# Step 2: Model Training
def train_models(df):
    # Tire degradation model
    X = df[['TyreLife', 'TrackAbrasiveness', 'Temperature']]
    y = df['LapTimeSeconds']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train, y_train)
    print(f"Random Forest R²: {rf_model.score(X_test, y_test):.2f}")

    # LSTM model
    sequence_length = 5
    X_seq, y_seq = [], []
    for i in range(len(df) - sequence_length):
        X_seq.append(df[['TyreLife', 'TrackAbrasiveness', 'Temperature']].iloc[i:i+sequence_length].values)
        y_seq.append(df['LapTimeSeconds'].iloc[i+sequence_length])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    lstm_model = Sequential([
        LSTM(50, input_shape=(sequence_length, 3)),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_seq, y_seq, epochs=20, batch_size=32, verbose=0)
    print("LSTM model trained")

    return rf_model, lstm_model

# Step 3: Strategy Optimization
def optimize_strategy(df, model):
    tire_properties = {
        'Soft': {'life': 20, 'pace': 1.0},
        'Medium': {'life': 30, 'pace': 1.02},
        'Hard': {'life': 40, 'pace': 1.05}
    }

    total_laps = 66
    best_strategy = []
    current_tire = 'Soft'
    current_age = 0
    total_time = 0
    pit_time = 20  # seconds

    # Ensure df is not empty
    if df.empty:
        print("Error: DataFrame is empty. Cannot optimize strategy.")
        return []

    for lap in range(total_laps):
        if current_age >= tire_properties[current_tire]['life']:
            best_strategy.append((lap, current_tire))
            current_tire = 'Medium' if current_tire == 'Soft' else 'Hard'
            total_time += pit_time
            current_age = 0

        # Predict lap time
        # Modulo to access temperature if lap exceeds DataFrame size
        temperature_index = lap % len(df)
        features = [[
            current_age,
            8,  # Track abrasiveness
            df['Temperature'].iloc[temperature_index]  # Access temperature with modulo
        ]]
        lap_time = model.predict(features)[0] * tire_properties[current_tire]['pace']
        total_time += lap_time
        current_age += 1

    print(f"Optimal strategy: {best_strategy}")
    print(f"Predicted total time: {total_time/60:.2f} minutes")
    return best_strategy

# Step 4: Dashboard
def create_dashboard(df, strategy):
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("F1 Race Strategy Analyzer"),
        dcc.Graph(id='degradation-plot'),
        html.Div([
            html.H3("Recommended Pit Stops"),
            html.Ul([html.Li(f"Lap {lap}: Change to {tire}") for lap, tire in strategy])
        ])
    ])

    @app.callback(
        Output('degradation-plot', 'figure'),
        Input('degradation-plot', 'id')
    )
    def update_plot(_):
        fig = px.line(df, x='LapNumber', y='LapTimeSeconds',
                     color='Compound', title='Tire Degradation Analysis',
                     labels={'LapTimeSeconds': 'Lap Time (seconds)'})
        return fig

    print("Dashboard ready at http://localhost:8050")
    return app

# Main execution
if __name__ == '__main__':
    # Data pipeline
    f1_data = fetch_data()

    if f1_data is not None:  # Proceed only if data was fetched successfully
        # Model training
        rf_model, lstm_model = train_models(f1_data)

        # Strategy optimization
        optimal_strategy = optimize_strategy(f1_data, rf_model)

        # Visualization
        app = create_dashboard(f1_data, optimal_strategy)
        app.run_server(debug=True)
    else:
        print("Failed to fetch data. Exiting.")
