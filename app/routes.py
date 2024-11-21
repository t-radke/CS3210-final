import pandas as pd
import joblib
from flask import render_template, jsonify, request
from app import app

# Load the pre-trained model
MODEL_FILE = "models/nfl_game_predictor.pkl"
try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    model = None
    print("Model not found. Ensure 'nfl_game_predictor.pkl' is in the 'models/' folder.")

@app.route('/')
def home():
    return render_template('index.html', title="NFL Game Prediction App")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Parse input data
        input_data = request.json
        required_features = [
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
        if not all(feature in input_data for feature in required_features):
            return jsonify({"error": "Missing required features"}), 400

        # Prepare features for prediction
        features = [[
            input_data['total_passing_yards'],
            input_data['total_rushing_yards'],
            input_data['passing_attempts'],
            input_data['rushing_attempts'],
            input_data['sacks_allowed'],
            input_data['interceptions_thrown'],
            input_data['fumbles_lost'],
            input_data['total_home_score'],
            input_data['total_away_score']
        ]]

        # Make prediction
        prediction = model.predict(features)[0]  # 0 = Loss, 1 = Win
        probability = model.predict_proba(features)[0].tolist()

        return jsonify({
            "prediction": int(prediction),
            "probability": probability  # [probability of loss, probability of win]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/predict_teams', methods=['POST'])
def predict_teams():
    try:
        input_data = request.json
        home_team = input_data.get('home_team')
        away_team = input_data.get('away_team')

        if not home_team or not away_team:
            return jsonify({"error": "Both teams must be specified"}), 400

        # Aggregate stats for the selected teams
        home_team_stats = data[data['posteam'] == home_team].mean()
        away_team_stats = data[data['posteam'] == away_team].mean()

        # Prepare features for prediction
        features = [[
            home_team_stats['total_passing_yards'],
            home_team_stats['total_rushing_yards'],
            home_team_stats['passing_attempts'],
            home_team_stats['rushing_attempts'],
            home_team_stats['sacks_allowed'],
            home_team_stats['interceptions_thrown'],
            home_team_stats['fumbles_lost'],
            home_team_stats['total_home_score'],
            away_team_stats['total_away_score']
        ]]

        # Make prediction for home team win probability
        probabilities = model.predict_proba(features)[0]
        home_win_probability = probabilities[1]  # Probability of home team winning

        # Determine the winner
        winner = home_team if home_win_probability > 0.5 else away_team

        return jsonify({
            "home_team": home_team,
            "away_team": away_team,
            "home_win_probability": home_win_probability,
            "winner": winner
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/current_games', methods=['GET'])
def current_games():
    try:
        schedule = get_current_week_schedule()
        games = schedule.to_dict(orient='records')
        return jsonify(games)
    except Exception as e:
        return jsonify({"error": str(e)}), 500