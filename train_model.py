import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import joblib
import os

if not os.path.exists('models'):
    os.makedirs('models')

#Load the updated dataset
DATA_FILE = "/Users/tylerradke/Documents/MSU_Denver/FALL_2024/CS3210/FinalProject/CS3210-final/data/Updated_Team_Specific_NFL_Game_Data.csv"
data = pd.read_csv(DATA_FILE)

#Add derived features
data['rush_attempt_diff'] = data['home_rush_attempt'] - data['away_rush_attempt']
data['pass_attempt_diff'] = data['home_pass_attempt'] - data['away_pass_attempt']
data['sack_diff'] = data['home_sack'] - data['away_sack']
data['passing_yards_diff'] = data['home_passing_yards'] - data['away_passing_yards']
data['rushing_yards_diff'] = data['home_rushing_yards'] - data['away_rushing_yards']
data['epa_diff'] = data['home_epa'] - data['away_epa']

#Select features (home, away, allowed stats, and derived stats) and target
X = data[
    [
        'home_rush_attempt', 'home_pass_attempt', 'home_sack',
        'home_passing_yards', 'home_rushing_yards', 'home_epa',
        'away_rush_attempt', 'away_pass_attempt', 'away_sack',
        'away_passing_yards', 'away_rushing_yards', 'away_epa',
        'rush_attempt_diff', 'pass_attempt_diff', 'sack_diff',
        'passing_yards_diff', 'rushing_yards_diff', 'epa_diff',
        'rushing_attempts_allowed', 'passing_attempts_allowed',
        'sacks_allowed', 'passing_yards_allowed', 'rushing_yards_allowed',
        'epa_allowed', 'qb_epa_allowed', 'xyac_epa_allowed',
    ]
]
y = data['win']  #Target column (1 for home team win, 0 for away team win)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Random Forest Model Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\nClassification Report:")
print(report)


joblib.dump(model, "models/nfl_game_predictor_updated.pkl")
print("Updated model saved to 'models/nfl_game_predictor_updated.pkl'")
