import nfl_data_py as nfl
import pandas as pd
import datetime

def get_current_week_schedule():
    try:
        #Load NFL schedules for the current season
        schedule = nfl.import_schedules([2024])  # Replace with the desired season

        #Define the start of the NFL season
        season_start = pd.Timestamp("2024-09-05")

        #Calculate the current week
        today = pd.Timestamp.now()
        days_since_start = (today - season_start).days
        current_week = (days_since_start // 7) + 1

        #Filter for games in the current week
        schedule['week'] = (pd.to_datetime(schedule['gameday']) - season_start).dt.days // 7 + 1
        filtered_schedule = schedule[schedule['week'] == current_week]

        team_data = pd.read_csv('data/Updated_Team_Specific_NFL_Game_Data.csv')  # Updated path
        merged_schedule = filtered_schedule.merge(
            team_data[['game_id', 'total_home_score', 'total_away_score']],
            on='game_id',
            how='left'
        )

        return merged_schedule[['game_id', 'home_team', 'away_team', 'gameday', 'total_home_score', 'total_away_score']].to_dict('records'), current_week
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return [], 0

