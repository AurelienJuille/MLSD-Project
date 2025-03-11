import pickle
import pandas as pd

from flask import Flask, request, render_template, jsonify, redirect, url_for

app = Flask(__name__)

"""
Prepare data for prediction

Args:
    df: data to predict 
    
Returns:
    to_predict: data to predict after feature selection
"""
def apply_feature_selection(original_features):
    selected_features_indices = [2, 11, 12, 16, 21, 24, 25, 27, 29, 30, 31, 32, 33, 35, 37] 
    return original_features.iloc[selected_features_indices]

"""
Predict the result of a game

Args: 
    to_predict: data to predict
    
Returns:
    prediction: prediction result 
"""
def predict(features):
    # to_predict["diffMinionsKilled"] = to_predict["blueTeamMinionsKilled"] - to_predict["redTeamMinionsKilled"]
    # to_predict["diffJungleMinions"] = to_predict["blueTeamJungleMinions"] - to_predict["redTeamJungleMinions"]
    # to_predict["diffTotalGold"] = to_predict["blueTeamTotalGold"] - to_predict["redTeamTotalGold"]
    # to_predict["diffTotalKills"] = to_predict["blueTeamTotalKills"] - to_predict["redTeamTotalKills"]
    # to_predict["diffXp"] = to_predict["blueTeamXp"] - to_predict["redTeamXp"]
    # to_predict["diffTotalDamageToChamps"] = to_predict["blueTeamTotalDamageToChamps"] - to_predict["redTeamTotalDamageToChamps"]
    # to_predict["diffDragonKills"] = to_predict["blueTeamDragonKills"] - to_predict["redTeamDragonKills"]
    # to_predict["diffHeraldKills"] = to_predict["blueTeamHeraldKills"] - to_predict["redTeamHeraldKills"]
    # to_predict["diffTowersDestroyed"] = to_predict["blueTeamTowersDestroyed"] - to_predict["redTeamTowersDestroyed"]
    # to_predict["diffInhibitorsDestroyed"] = to_predict["blueTeamInhibitorsDestroyed"] - to_predict["redTeamInhibitorsDestroyed"]
    # to_predict["diffTurretPlatesDestroyed"] = to_predict["blueTeamTurretPlatesDestroyed"] - to_predict["redTeamTurretPlatesDestroyed"]

    to_predict = apply_feature_selection(features) 
    to_predict = to_predict.values.reshape(1, -1)
    
    loaded_model = pickle.load(open("model.pkl", "rb"))
    prediction = loaded_model.predict(to_predict)

    return prediction[0]

"""
Predict the result of a match

Args:
    match_id: id of the match to predict

Returns:
    prediction: prediction result (1 if blue team wins, 0 if red team wins) or -1 if match not found
"""
def predict_match(match_id):
    
    # TODO: implement this function with the riot API
    
    df = pd.read_csv("data/processed_dataset.csv", index_col=0) # ONLY FOR TESTING
    df = df.drop(columns=["blueWin"])
    
    try:
        match_features = df.loc[match_id]
    except KeyError:
        return -1 # Match not found  
    
    return predict(match_features)

@app.route('/predict', methods=['POST'])
def get_prediction():
    match_id = request.form['matchId']
    prediction = predict_match(match_id)
    
    if prediction == -1:
        return jsonify({"error": "Match not found"})
    elif prediction == 1:
        return jsonify({"prediction": "Blue team wins"})
    else:
        return jsonify({"prediction": "Red team wins"})

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, port=8080)