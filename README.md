# **NFL Game Predictor**

## **Overview**
The NFL Game Predictor is a web application designed to predict the outcomes of NFL games using machine learning. It provides users with the ability to view the weekly schedule, predict game winners, and visualize win probabilities through pie charts. The predictions are based on a Random Forest model trained on key game metrics.

---

## **Features**
- Predict game outcomes using a Random Forest Classifier.
- Display win probabilities 
- Provide explanations for metrics
- Interactive weekly schedule for selecting games.



## **Project Structure**

```
app
|__static
|____probability_charts
|__templates
|____init__.py
|__nfl_schedule.py
|__routes.py
data
|__current_season_data.csv
|__Updated_Team_Specific_NFL_Games.csv
models
|__train_model.py
|__nfl_game_predictor_updated.pkl
Project-Milestone-OPEN-ME
|__current_season_data.csv
|__project_milestone_radke.ipynb
|__team_game_stats.csv
.gitignore
README.md
requirements.txt
run.py
```

# Instructions 

### **Installation**
Clone the repository and open in VSCode:
```
git clone https://github.com/t-radke/CS3210-final.git
```
initialize virtual environment and install requirements.txt

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### **Setup**
The project includes a pre-trained model (`nfl_game_predictor_updated.pkl`) in the `models/` directory. You can skip the model training step unless you want to retrain or update the model with new data.

To retrain the model (optional) navigate to `models/` folder and run:
```
python train_model.py
```
Verify that the updated model file (`nfl_game_predictor_updated.pkl`) is in the `models/` folder.

### **Running the Application**

Start the Flask application by running:

```
python run.py
```

## **Model and Data Overview**

### **Model**
This project uses a **Random Forest Classifier** to predict the outcome of NFL games. The Random Forest model is well-suited for this project due to its ability to handle large datasets with mixed types of features and capture complex interactions between variables. Here are the details:

- **Features Used**:
  - Offensive stats: Rushing attempts, passing attempts, rushing yards, passing yards, sacks, EPA (Expected Points Added)
  - Defensive stats: Passing yards allowed, rushing yards allowed, sacks allowed, EPA allowed
  - Derived features: Differences in rushing and passing yards, sacks, EPA

- **Target Variable**: The model predicts whether the **home team** will win (`1`) or lose (`0`).

- **Performance**: The model was trained and tested with an **80-20 split**, achieving an accuracy of **91%** on the test set.

- **File**: The trained model is saved as `models/nfl_game_predictor_updated.pkl`.

---

### **Data Collection and Processing**
#### **Raw Data Source: NFLfastr**
- **NFLfastr**: A robust library for retrieving NFL play-by-play data. It provides detailed game-level and team-level statistics for all NFL seasons.
- `nflfastr` Python library was utlizied to scrape a comprehensive dataset of NFL games and team performances. 

#### **Data Cleaning and Feature Engineering**
1. **Raw Data Cleaning**:
   - Removed irrelevant columns and games without sufficient data (e.g., missing EPA values).
   - Handled missing values and standardized column names for consistency.

2. **Derived Features**:
   - Added features like `rush_attempt_diff`, `pass_attempt_diff`, `sack_diff`, and `epa_diff` to capture key statistical differences between teams.
   - Aggregated team statistics into season averages to better reflect overall performance.

3. **Final Dataset**:
   - The cleaned dataset is stored as `data/Updated_Team_Specific_NFL_Game_Data.csv` 

---

### **Schedule Retrieval**
The weekly schedule of NFL games is dynamically fetched and displayed in the application.

1. **Implementation**:
   - The schedule data is not manually inputted but dynamically retrieved using a script.
   - The function `get_current_week_schedule()` in `app/nfl_schedule.py` ensures the app always shows the latest week's games.

2. **Process**:
   - The schedule includes home and away teams, the game date, and a unique identifier (`game_id`) used for predictions.
   - This ensures the schedule adapts to new NFL seasons automatically.

3. **Flexibility**:
   - The schedule integrates seamlessly with the prediction system, allowing users to select any game and view predictions for that matchup.

---

## **Limitations**

1. **Static Dataset**:
   - The dataset used for predictions only includes statistics through a specific week of the NFL season.
   - To update predictions for future weeks or the next season, a new dataset must be fetched, cleaned, and integrated manually. This means the app does not currently support real-time updates.

2. **Team-Specific Variability**:
   - The model's predictions rely on aggregated team statistics rather than game-specific data. This can miss nuances such as:
     - Injuries to key players
     - Mid-season trades
     - Weather conditions

3. **Limited Feature Set**:
   - While the features chosen (e.g., rushing yards, passing yards, EPA) are strong predictors of game outcomes, additional advanced metrics (e.g., play-by-play data or player-specific stats) could improve prediction accuracy.


## **Model Update Note**

Please note that the current model (`nfl_game_predictor_updated.pkl`) has been updated from the original project milestone version. This new model incorporates additional features, improved data preprocessing, and updated statistics, ensuring higher accuracy and better performance.

### **Key Differences from the Milestone Model**:
1. **Additional Features**:
   - New derived features like rushing and passing attempt differentials, EPA differentials, and more were added to enhance predictive power.

2. **Improved Dataset**:
   - The dataset was cleaned and updated to include more accurate team-specific statistics up to the specific week used in this project.





