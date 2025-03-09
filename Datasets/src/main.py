import fastf1 as ff1

# Define the year for which you want to get the event names
year = 2024

# Get the event schedule for the specified year
event_schedule = ff1.get_event_schedule(year)

# Print the event names
event_names = event_schedule['EventName']
for name in event_names:
    print(name)
