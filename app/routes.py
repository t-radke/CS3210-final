import pandas as pd
from flask import render_template, jsonify
from app import app

# Load the dataset
DATA_FILE = "data/current_season_data.csv"  # Update the path if needed

@app.route('/games')
def games():
    try:
        # Load the dataset
        data = pd.read_csv(DATA_FILE)
        
        # Convert the first 10 rows to a dictionary for display
        games_preview = data.head(10).to_dict(orient='records')
        
        # Return as JSON for simplicity (you can use a template later)
        return jsonify(games_preview)
    except Exception as e:
        return f"Error loading data: {e}"
