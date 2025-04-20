"""
This module implements a Flask web application for predicting the outcome of
League of Legends matches in real-time using a machine learning model.

It integrates with the Riot Games API to fetch live game data, processes the
data into features suitable for the model, and provides predictions along with
explanations using LIME (Local Interpretable Model-agnostic Explanations).

Key functionalities:
- Fetch live game data from the Riot API.
- Extract and preprocess features for prediction.
- Use a pre-trained machine learning model to predict the probability of the
    blue team winning.
- Provide explanations for predictions using LIME.
- Maintain a history of predictions and allow updates to past predictions.

Routes:
- `/predict`: Predict the outcome of the current match.
- `/past_predictions`: Retrieve or update the history of predictions.
- `/`: Render the main application interface.

Dependencies:
- Flask for the web application.
- Google Cloud AI Platform for model storage and retrieval.
- LIME for generating explanations.
- Riot Games API for live game data.
"""

import tempfile

import joblib
import lime
import lime.lime_tabular
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
from google.cloud import aiplatform, storage

app = Flask(__name__)

history = []

aiplatform.init(
    project="lolffate", location="europe-west1"
)  # Initialize the AI Platform project and location

models = aiplatform.Model.list(
    filter='display_name="lolffate"'
)  # Get the list of models with the display name "lolffate"
latest_model = sorted(models, key=lambda m: m.create_time, reverse=True)[
    0
]  # Get the latest model


def load_artifact_from_gcs(gcs_uri: str, filename: str):
    """
    Downloads a file from GCS and loads it with joblib.

    Args:
        gcs_uri: The GCS URI to the folder containing the artifact
        filename: The name of the file to download

    Returns:
        The loaded Python object
    """
    storage_client = storage.Client()
    bucket_name, blob_prefix = gcs_uri.replace("gs://", "").split("/", 1)
    blob_path = f"{blob_prefix}/{filename}"

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        blob.download_to_filename(temp_file.name)
        obj = joblib.load(temp_file.name)

    return obj


model = load_artifact_from_gcs(
    latest_model.uri, "model.joblib"
)  # Load the model from GCS once at startup


def create_global_explainer():
    """
    Create a global LIME explainer for the model.
    This function loads the dataset and initializes the explainer.
    """
    # Load the dataset
    df_rep = pd.read_csv("data/processed_dataset.csv", index_col=0).drop(
        columns=["blueWin"]
    )

    # Apply feature selection to each row and create a numpy array
    X_train = df_rep.apply(lambda row: row, axis=1)
    X_train = X_train.to_numpy()

    # Extract feature names
    feature_names = list(df_rep.iloc[0].index)

    # Create and return the explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["Red team", "Blue team"],
        mode="classification",
    )
    return explainer


# Create the global explainer once, at startup
global_explainer = create_global_explainer()


def predict(features):
    """
    Predict the result of a game

    Args:
        to_predict: data to predict

    Returns:
        prediction: prediction result (probability of blue team winning, float [0, 1])
    """
    # to_predict["diffMinionsKilled"] = (
    #     to_predict["blueTeamMinionsKilled"] - to_predict["redTeamMinionsKilled"]
    # )
    # to_predict["diffJungleMinions"] = (
    #     to_predict["blueTeamJungleMinions"] - to_predict["redTeamJungleMinions"]
    # )
    # to_predict["diffTotalGold"] = (
    #     to_predict["blueTeamTotalGold"] - to_predict["redTeamTotalGold"]
    # )
    # to_predict["diffTotalKills"] = (
    #     to_predict["blueTeamTotalKills"] - to_predict["redTeamTotalKills"]
    # )
    # to_predict["diffXp"] = (
    #     to_predict["blueTeamXp"] - to_predict["redTeamXp"]
    # )
    # to_predict["diffTotalDamageToChamps"] = (
    #     to_predict["blueTeamTotalDamageToChamps"]
    #     - to_predict["redTeamTotalDamageToChamps"]
    # )
    # to_predict["diffDragonKills"] = (
    #     to_predict["blueTeamDragonKills"] - to_predict["redTeamDragonKills"]
    # )
    # to_predict["diffHeraldKills"] = (
    #     to_predict["blueTeamHeraldKills"] - to_predict["redTeamHeraldKills"]
    # )
    # to_predict["diffTowersDestroyed"] = (
    #     to_predict["blueTeamTowersDestroyed"]
    #     - to_predict["redTeamTowersDestroyed"]
    # )
    # to_predict["diffInhibitorsDestroyed"] = (
    #     to_predict["blueTeamInhibitorsDestroyed"]
    #     - to_predict["redTeamInhibitorsDestroyed"]
    # )
    # to_predict["diffTurretPlatesDestroyed"] = (
    #     to_predict["blueTeamTurretPlatesDestroyed"]
    #     - to_predict["redTeamTurretPlatesDestroyed"]
    # )

    to_predict = features.values.reshape(1, -1)

    prediction = model.predict_proba(to_predict)[
        0, 1
    ]  # Probability of blue team winning for the first sample

    return prediction


def extract_game_data(data):
    """
    Extracts structured features from the in-game Riot API data.

    Args:
        data (dict): raw JSON data from /liveclientdata/allgamedata

    Returns:
        dict: feature set ready for model input
    """
    players = data["allPlayers"]
    events = data.get("events", {}).get("Events", [])
    blue_team = [p for p in players if p["team"] == "ORDER"]
    red_team = [p for p in players if p["team"] == "CHAOS"]

    def team_stats(team):
        return {
            "kills": sum(p["scores"]["kills"] for p in team),
            "wards": sum(p["scores"].get("wardScore", 0) for p in team),
            "gold": sum(p.get("currentGold", 0) for p in team),
            "minions": sum(p["scores"]["creepScore"] for p in team),
            "jungle_minions": sum(
                p["scores"].get("neutralMinionsKilled", 0) for p in team
            ),
            "xp": sum(p["level"] for p in team),
            "dmg_champs": sum(
                p["scores"].get("damageDealtToChampions", 0) for p in team
            ),
        }

    blue = team_stats(blue_team)
    red = team_stats(red_team)

    # Process timeline events
    first_blood = 0
    heralds = {"ORDER": 0, "CHAOS": 0}
    dragons = {"ORDER": 0, "CHAOS": 0}
    turrets = {"ORDER": 0, "CHAOS": 0}
    inhibitors = {"ORDER": 0, "CHAOS": 0}
    turret_plates = {"ORDER": 0, "CHAOS": 0}

    for event in events:
        if event["EventName"] == "FirstBlood":
            recipient = event["Recipient"]
            if any(
                p["summonerName"] == recipient and p["team"] == "ORDER" for p in players
            ):
                first_blood = 1
        elif event["EventName"] == "HeraldKill":
            killer = event["KillerName"]
            team = next(
                (p["team"] for p in players if p["summonerName"] == killer), None
            )
            if team:
                heralds[team] += 1
        elif event["EventName"] == "DragonKill":
            killer = event["KillerName"]
            team = next(
                (p["team"] for p in players if p["summonerName"] == killer), None
            )
            if team:
                dragons[team] += 1
        elif event["EventName"] == "TurretKilled":
            killer = event["KillerName"]
            team = next(
                (p["team"] for p in players if p["summonerName"] == killer), None
            )
            if team:
                turrets[team] += 1
        elif event["EventName"] == "BuildingKill":
            building_type = event.get("BuildingType", "")
            if "Inhibitor" in building_type:
                killer = event["KillerName"]
                team = next(
                    (p["team"] for p in players if p["summonerName"] == killer), None
                )
                if team:
                    inhibitors[team] += 1
        elif event["EventName"] == "TurretPlateDestroyed":
            killer = event["KillerName"]
            team = next(
                (p["team"] for p in players if p["summonerName"] == killer), None
            )
            if team:
                turret_plates[team] += 1

    features = {
        "blueTeamTotalKills": blue["kills"],
        "redTeamTotalKills": red["kills"],
        "blueTeamDragonKills": dragons["ORDER"],
        "redTeamDragonKills": dragons["CHAOS"],
        "blueTeamControlWardsPlaced": 0,  # Not available in the API
        "blueTeamWardsPlaced": blue["wards"],
        "blueTeamHeraldKills": heralds["ORDER"],
        "blueTeamTowersDestroyed": turrets["ORDER"],
        "blueTeamInhibitorsDestroyed": inhibitors["ORDER"],
        "blueTeamTurretPlatesDestroyed": turret_plates["ORDER"],
        "blueTeamFirstBlood": first_blood,
        "blueTeamMinionsKilled": blue["minions"],
        "blueTeamJungleMinions": blue["jungle_minions"],
        "blueTeamTotalGold": blue["gold"],
        "blueTeamXp": blue["xp"],
        "blueTeamTotalDamageToChamps": blue["dmg_champs"],
        "redTeamControlWardsPlaced": 0,
        "redTeamWardsPlaced": red["wards"],
        "redTeamHeraldKills": heralds["CHAOS"],
        "redTeamTowersDestroyed": turrets["CHAOS"],
        "redTeamInhibitorsDestroyed": inhibitors["CHAOS"],
        "redTeamTurretPlatesDestroyed": turret_plates["CHAOS"],
        "redTeamMinionsKilled": red["minions"],
        "redTeamJungleMinions": red["jungle_minions"],
        "redTeamTotalGold": red["gold"],
        "redTeamXp": red["xp"],
        "redTeamTotalDamageToChamps": red["dmg_champs"],
        # Diffs
        "diffMinionsKilled": blue["minions"] - red["minions"],
        "diffJungleMinions": blue["jungle_minions"] - red["jungle_minions"],
        "diffTotalGold": blue["gold"] - red["gold"],
        "diffTotalKills": blue["kills"] - red["kills"],
        "diffXp": blue["xp"] - red["xp"],
        "diffTotalDamageToChamps": blue["dmg_champs"] - red["dmg_champs"],
        "diffDragonKills": dragons["ORDER"] - dragons["CHAOS"],
        "diffHeraldKills": heralds["ORDER"] - heralds["CHAOS"],
        "diffTowersDestroyed": turrets["ORDER"] - turrets["CHAOS"],
        "diffInhibitorsDestroyed": inhibitors["ORDER"] - inhibitors["CHAOS"],
        "diffTurretPlatesDestroyed": turret_plates["ORDER"] - turret_plates["CHAOS"],
    }

    return features


def predict_match():
    """
    Predict the result of the current in-game match using the local Riot API

    Returns:
        prediction: probability of blue team winning
        exp: explanation (LIME)
    """
    try:
        response = requests.get(
            "https://127.0.0.1:2999/liveclientdata/allgamedata", verify=False
        )

        if response.status_code != 200:
            return -1, None

        data = response.json()

    except Exception:
        return -1, None

    try:
        features = extract_game_data(data)

        df = pd.DataFrame([features])
        match_features = df.iloc[0]

        prediction = predict(match_features)

        exp = global_explainer.explain_instance(
            match_features.values, model.predict_proba, num_features=5
        )

        local_player = next(
            (
                p
                for p in data["allPlayers"]
                if p["summonerName"] == data["activePlayer"]["summonerName"]
            ),
            None,
        )
        player_team = (
            "blue" if local_player and local_player["team"] == "ORDER" else "red"
        )

        return prediction, exp, player_team

    except Exception as e:
        return -1, f"Error processing match data: {str(e)}"


def get_history_string():
    """
    Generates a formatted HTML string summarizing the match history and predictions.
    The function iterates through a list of match history records, where each record
    contains a match ID and a prediction value. Based on the prediction value, it
    determines the winning team (Blue or Red) and appends a formatted string to the
    result. The prediction is expressed as a percentage.
    Returns:
        str: An HTML-formatted string containing the match results and probabilities.
    """

    history_string = ""
    for h in history:
        if h["prediction"] >= 0.5:
            history_string += (
                f"Match {h['match_id']}: Blue team wins with probability of "
                + str(round(h["prediction"] * 100, 2))
                + "%<br>"
            )
        else:
            history_string += (
                f"Match {h['match_id']}: Red team wins with probability of "
                + str(round((1 - h["prediction"]) * 100, 2))
                + "%<br>"
            )
    return history_string


@app.route("/past_predictions/last", methods=["PUT"])
def update_last_history():
    """
    Updates the last prediction in the history with a new value provided in the request.
    Returns:
        Response: A JSON response containing the updated history or an error message
                  if the history is empty.
    """

    if len(history) == 0:
        return jsonify({"error": "No prediction history found"})

    history[-1]["prediction"] = float(request.form["update"])

    return jsonify({"history": get_history_string()})


@app.route("/past_predictions/<id>", methods=["PUT"])
def update_history(id):
    """
    Updates the prediction value for a specific match in the history.
    Args:
        id (int): The match ID to update the prediction for.
    Returns:
        Response: A JSON response containing the updated history or an error message.
    """

    if len(history) == 0:
        return jsonify({"error": "No prediction history found"})

    entry = next((h for h in history if h["match_id"] == id), None)
    if not entry:
        return jsonify({"error": "Prediction not found"})

    entry["prediction"] = float(request.form["update"])

    return jsonify({"history": get_history_string()})


@app.route("/past_predictions", methods=["GET"])
def get_history():
    """
    Retrieve and return the history as a JSON response.
    Returns:
        Response: A JSON object containing the history string or an error message.
    """

    try:
        return jsonify({"history": get_history_string()})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/predict", methods=["POST"])
def get_prediction():
    """
    Handle POST requests to predict the outcome of the current match.
    """
    prediction, exp, player_team = predict_match()
    print("prediction:", prediction)

    if prediction == -1:
        return jsonify({"error": "Match not found"})

    explanation_list = exp.as_list()
    explanation_html = (
        """
	                   <div style='text-align: center; margin-top: 10px;'>
	                   <table style='font-family: monospace; text-align: left; display: inline-block;'>
	                   """
        + "\n".join(
            [
                f"<tr>"
                f"<td>{feat}</td>"
                f"<td style='padding: 0 2em;'>→</td>"
                f"<td style='color:{'limegreen' if weight > 0 else 'tomato'};'>"
                f"{'+' if weight > 0 else ''}{weight * 100:.2f}%"
                f"</td>"
                f"</tr>"
                for feat, weight in explanation_list
            ]
        )
        + """
	     </table>
	     </div>
	     """
    )

    # Return the probability for blue team winning (a float between 0 and 1)
    return jsonify(
        {
            "probability": prediction,
            "explanation": explanation_html,
            "player_team": player_team,
        }
    )


@app.route("/")
def index():
    """
    Render the index.html template for the home page.
    """

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=8080)
