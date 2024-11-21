import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Ensure the models folder exists
if not os.path.exists('models'):
    os.makedirs('models')

# Load dataset
DATA_FILE = "data/Aggregated_Game-Level_Data.csv"
data = pd.read_csv(DATA_FILE)

# Derive the target column: Win
data['Win'] = (data['total_home_score'] > data['total_away_score']).astype(int)

# Select features and target
X = data[
    [
        'total_passing_yards',
        'total_rushing_yards',
        'passing_attempts',
        'rushing_attempts',
        'sacks_allowed',
        'interceptions_thrown',
        'fumbles_lost',
        'total_home_score',
        'total_away_score'
    ]
]
y = data['Win']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Random Forest Model Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, "models/nfl_game_predictor.pkl")
print("Model saved to 'models/nfl_game_predictor.pkl'")


