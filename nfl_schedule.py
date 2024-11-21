import nfl_data_py as nfl
import pandas as pd
import datetime

def get_current_week_schedule():
    # Load schedule for the current season
    schedule = nfl.import_schedules([2024])  # Replace with the desired season

    # Get today's date and filter for the current week
    today = datetime.date.today()
    current_week = schedule[
        (pd.to_datetime(schedule['game_date']) >= today - datetime.timedelta(days=7)) &
        (pd.to_datetime(schedule['game_date']) <= today)
    ]

    # Return the current week's games
    return current_week[['game_id', 'home_team', 'away_team', 'game_date']]
