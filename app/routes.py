from app import app
from flask import render_template, request, url_for
from app.nfl_schedule import get_current_week_schedule
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import joblib, os
import pandas as pd
import numpy as np

# Load the trained model and data
model = joblib.load("models/nfl_game_predictor_updated.pkl")
DATA_FILE = "data/Updated_Team_Specific_NFL_Game_Data.csv"
team_data = pd.read_csv(DATA_FILE)

@app.route('/')
def home():
    return render_template('index.html')
from datetime import date

@app.route('/schedule')
def schedule():
    # Get games and the current week
    games, current_week = get_current_week_schedule()

    # Debug: Print games and week to verify
    print(f"Games Passed to Template (Week {current_week}):")
    for game in games:
        print(game)

    # Pass games and the current week to the template
    return render_template('schedule.html', games=games, current_week=current_week)

@app.route('/predict/<game_id>', methods=['GET'])
def predict(game_id):
    # Parse game_id to get the home and away teams
    try:
        home_team = game_id.split('_')[2]
        away_team = game_id.split('_')[3]
    except IndexError:
        return "Invalid game ID.", 404

    # Aggregate stats for the home and away teams
    try:
        home_stats = team_data[
            (team_data['game_id'].str.contains(home_team))
        ].mean(numeric_only=True)

        away_stats = team_data[
            (team_data['game_id'].str.contains(away_team))
        ].mean(numeric_only=True)
    except Exception as e:
        return f"Error aggregating stats: {e}", 500

    # Check if stats were found for both teams
    if home_stats.empty or away_stats.empty:
        return f"Error: Stats not found for one or both teams (Home: {home_team}, Away: {away_team}).", 404

    # Dynamically calculate derived features
    rush_attempt_diff = home_stats['home_rush_attempt'] - away_stats['away_rush_attempt']
    pass_attempt_diff = home_stats['home_pass_attempt'] - away_stats['away_pass_attempt']
    sack_diff = home_stats['home_sack'] - away_stats['away_sack']
    passing_yards_diff = home_stats['home_passing_yards'] - away_stats['away_passing_yards']
    rushing_yards_diff = home_stats['home_rushing_yards'] - away_stats['away_rushing_yards']
    epa_diff = home_stats['home_epa'] - away_stats['away_epa']

    # Create input data for the model
    input_data = {
        'home_rush_attempt': home_stats['home_rush_attempt'],
        'home_pass_attempt': home_stats['home_pass_attempt'],
        'home_sack': home_stats['home_sack'],
        'home_passing_yards': home_stats['home_passing_yards'],
        'home_rushing_yards': home_stats['home_rushing_yards'],
        'home_epa': home_stats['home_epa'],
        'away_rush_attempt': away_stats['away_rush_attempt'],
        'away_pass_attempt': away_stats['away_pass_attempt'],
        'away_sack': away_stats['away_sack'],
        'away_passing_yards': away_stats['away_passing_yards'],
        'away_rushing_yards': away_stats['away_rushing_yards'],
        'away_epa': away_stats['away_epa'],
        'rush_attempt_diff': rush_attempt_diff,
        'pass_attempt_diff': pass_attempt_diff,
        'sack_diff': sack_diff,
        'passing_yards_diff': passing_yards_diff,
        'rushing_yards_diff': rushing_yards_diff,
        'epa_diff': epa_diff,
        'rushing_attempts_allowed': home_stats['rushing_attempts_allowed'],
        'passing_attempts_allowed': home_stats['passing_attempts_allowed'],
        'sacks_allowed': home_stats['sacks_allowed'],
        'passing_yards_allowed': home_stats['passing_yards_allowed'],
        'rushing_yards_allowed': home_stats['rushing_yards_allowed'],
        'epa_allowed': home_stats['epa_allowed'],
        'qb_epa_allowed': home_stats['qb_epa_allowed'],
        'xyac_epa_allowed': home_stats['xyac_epa_allowed'],
        'rushing_attempts_allowed_away': away_stats['rushing_attempts_allowed'],
        'passing_attempts_allowed_away': away_stats['passing_attempts_allowed'],
        'sacks_allowed_away': away_stats['sacks_allowed'],
        'passing_yards_allowed_away': away_stats['passing_yards_allowed'],
        'rushing_yards_allowed_away': away_stats['rushing_yards_allowed'],
        'epa_allowed_away': away_stats['epa_allowed'],
        'qb_epa_allowed_away': away_stats['qb_epa_allowed'],
        'xyac_epa_allowed_away': away_stats['xyac_epa_allowed']
    }

    # Ensure the feature order matches the model
    input_df = pd.DataFrame([input_data])[model.feature_names_in_]

    # Predict the outcome using the model
    prediction = model.predict(input_df)[0]
    winning_team = home_team if prediction == 1 else away_team
    result = f"{winning_team} Wins"

        # Focused Key Factors for the Predicted Team
    factors = []

    if prediction == 1:  # Home team predicted to win
        # Rushing Attack
        if rush_attempt_diff > 2:
            factors.append(
                f"{home_team}'s rushing attack is strong, with {rush_attempt_diff:.1f} more attempts per game, likely controlling the game's tempo."
            )
        if away_stats.get('rushing_yards_allowed', 0) > home_stats.get('home_rushing_yards', 0):
            factors.append(
                f"{away_team} struggles to stop the run, conceding {away_stats['rushing_yards_allowed']:.1f} yards per game, which {home_team} can exploit."
            )

        # Passing Efficiency
        if passing_yards_diff > 50:
            factors.append(
                f"{home_team} excels in the passing game, with a {passing_yards_diff:.1f}-yard advantage, creating scoring opportunities."
            )
        if home_stats.get('passing_yards_allowed', 0) < 200:
            factors.append(
                f"{home_team}'s defense is exceptional against the pass, allowing only {home_stats['passing_yards_allowed']:.1f} yards per game."
            )

        # Defensive Strengths
        if epa_diff > 1.0:
            factors.append(
                f"{home_team}'s overall efficiency (EPA) reflects their ability to capitalize on key plays, a decisive factor in this game."
            )
        if sack_diff > 1:
            factors.append(
                f"{home_team}'s defensive line, with {sack_diff:.1f} more sacks per game, could disrupt {away_team}'s quarterback."
            )

        # Weaknesses of the Opponent
        if away_stats.get('passing_yards_allowed', 0) > 250:
            factors.append(
                f"{away_team} struggles against high-performing passing teams, allowing {away_stats['passing_yards_allowed']:.1f} yards per game."
            )
        if away_stats.get('sacks_allowed', 0) > 3:
            factors.append(
                f"{away_team}'s offensive line struggles to protect the quarterback, giving up {away_stats['sacks_allowed']:.1f} sacks per game."
            )

    else:  # Away team predicted to win
        # Rushing Attack
        if rush_attempt_diff < -2:
            factors.append(
                f"{away_team}'s ground game is dominant, with {abs(rush_attempt_diff):.1f} more attempts per game, likely wearing down {home_team}'s defense."
            )
        if home_stats.get('rushing_yards_allowed', 0) > away_stats.get('away_rushing_yards', 0):
            factors.append(
                f"{home_team} has difficulty stopping the run, conceding {home_stats['rushing_yards_allowed']:.1f} yards per game, which {away_team} can exploit."
            )

        # Passing Efficiency
        if passing_yards_diff < -50:
            factors.append(
                f"{away_team} thrives in the passing game, outgaining opponents by {abs(passing_yards_diff):.1f} yards per game."
            )
        if away_stats.get('passing_yards_allowed', 0) < 200:
            factors.append(
                f"{away_team}'s secondary is elite, holding opponents to only {away_stats['passing_yards_allowed']:.1f} yards per game."
            )

        # Defensive Strengths
        if epa_diff < -1.0:
            factors.append(
                f"{away_team}'s efficiency (EPA) gives them an edge, highlighting their ability to win critical moments in games."
            )
        if sack_diff < -1:
            factors.append(
                f"{away_team}'s pass rush, with {abs(sack_diff):.1f} more sacks per game, could disrupt {home_team}'s quarterback."
            )

        # Weaknesses of the Opponent
        if home_stats.get('passing_yards_allowed', 0) > 250:
            factors.append(
                f"{home_team} struggles against effective passing teams, allowing {home_stats['passing_yards_allowed']:.1f} yards per game."
            )
        if home_stats.get('sacks_allowed', 0) > 3:
            factors.append(
                f"{home_team}'s offensive line has trouble protecting the quarterback, allowing {home_stats['sacks_allowed']:.1f} sacks per game."
            )

    # Add a default explanation if no factors are generated
    if not factors:
        factors.append(
            f"The prediction is based on {winning_team}'s superior performance across key metrics this season."
        )

    input_df = pd.DataFrame([input_data])[model.feature_names_in_]

    # Predict win probability
    probabilities = model.predict_proba(input_df)[0]
    home_prob = probabilities[1]  # Probability for the home team
    away_prob = probabilities[0]  # Probability for the away team

    # Generate the win probability chart
    probability_chart_path = os.path.join("app/static/probability_charts", f"{game_id}_probability_chart.png")
    os.makedirs("app/static/probability_charts", exist_ok=True)
    generate_probability_chart(home_team, away_team, home_prob, away_prob, probability_chart_path)

    # Render the prediction result with probability chart
    return render_template(
        'predict.html',
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        prediction=f"{home_team} Wins" if home_prob > away_prob else f"{away_team} Wins",
        probability_chart_url=url_for('static', filename=f"probability_charts/{game_id}_probability_chart.png"),
        factors=factors
    )


def generate_probability_chart(home_team, away_team, home_prob, away_prob, save_path):

    # Pie chart data
    labels = [home_team, away_team]
    sizes = [home_prob * 100, away_prob * 100]
    colors = ['#1f77b4', '#ff7f0e']  # Blue for home, orange for away

    # Adjust figure size
    fig, ax = plt.subplots(figsize=(4, 4))  # Smaller size
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 10},
    )

    # Add a legend
    ax.legend(
        wedges,
        [f"{team} ({prob:.1f}%)" for team, prob in zip(labels, sizes)],
        title="Teams",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9,
    )

    # Title
    ax.set_title("Win Probability", fontsize=14)

    # Save the chart
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
