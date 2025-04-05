import pickle
import pandas as pd
import lime
import lime.lime_tabular

from flask import Flask, request, render_template, jsonify, redirect, url_for

app = Flask(__name__)

history = []



def apply_feature_selection(original_features):
    """
    Prepare data for prediction

    Args:
        df: data to predict 
        
    Returns:
        to_predict: data to predict after feature selection
    """
    #selected_features_indices = [2, 11, 12, 16, 21, 24, 25, 27, 29, 30, 31, 32, 33, 35, 37] 
    #return original_features.iloc[selected_features_indices]
    return original_features


def create_global_explainer():
    # Load the dataset
    df_rep = pd.read_csv("data/processed_dataset.csv", index_col=0).drop(columns=["blueWin"])
    
    # Apply feature selection to each row and create a numpy array
    X_train = df_rep.apply(lambda row: apply_feature_selection(row), axis=1)
    X_train = X_train.to_numpy()

    # Scale the data
    scaler = pickle.load(open("scaler.pkl", "rb"))
    X_train = scaler.transform(X_train)
    
    # Extract feature names
    feature_names = list(apply_feature_selection(df_rep.iloc[0]).index)
    
    # Create and return the explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["Red team", "Blue team"],
        mode='classification'
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

    #to_predict = apply_feature_selection(features)
    to_predict = features.values.reshape(1, -1)

    scaler = pickle.load(open("scaler.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))

    to_predict = scaler.transform(to_predict) # Scale the data

    prediction = model.predict_proba(to_predict)[0, 1] # Probability of blue team winning for the first sample

    return prediction


def predict_match(match_id):
    """
    Predict the result of a match

    Args:
        match_id: id of the match to predict

    Returns:
        prediction: prediction result (probability of blue team winning, float [0, 1]) or -1 if match not found
        exp: explanation of the prediction using LIME
    """

    # TODO: implement this function with the riot API
    
    df = pd.read_csv("data/processed_dataset.csv", index_col=0) # ONLY FOR TESTING
    df = df.drop(columns=["blueWin"])
    
    try:
        match_features = df.loc[match_id]
    except KeyError:
        return -1 # Match not found  
    
    prediction = predict(match_features)
    
    history.append({
        "match_id": match_id,
        "prediction": prediction
    })

    # Get the explanation for the prediction
    model = pickle.load(open("model.pkl", "rb"))
    exp = global_explainer.explain_instance(match_features.values, model.predict_proba, num_features=5)
    
    return prediction, exp

def get_history_string():
    history_string = ""
    for h in history:
        if h['prediction'] >= 0.5:
            history_string += f"Match {h['match_id']}: Blue team wins with probability of " + str(round(h['prediction'] * 100, 2)) + "%<br>"
        else:
            history_string += f"Match {h['match_id']}: Red team wins with probability of " + str(round((1-h['prediction']) * 100, 2)) + "%<br>"
    return history_string


@app.route("/past_predictions/last", methods=["PUT"])
def update_last_history():
    if len(history) == 0:
        return jsonify({"error": "No prediction history found"})
    
    history[-1]["prediction"] = float(request.form['update'])
    
    return jsonify({"history": get_history_string()})


@app.route("/past_predictions/<id>", methods=["PUT"])
def update_history(id):
    if len(history) == 0:
        return jsonify({"error": "No prediction history found"})
     
    entry = next((h for h in history if h["match_id"] == id), None)
    if not entry:
        return jsonify({"error": "Prediction not found"})
    
    entry["prediction"] = float(request.form['update'])
    
    return jsonify({"history": get_history_string()})


@app.route("/past_predictions", methods=["GET"]) 
def get_history():
    try:
        return jsonify({
            "history": get_history_string()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})
    

@app.route('/predict', methods=['POST'])
def get_prediction():
    match_id = request.form['matchId']
    prediction, exp = predict_match(match_id)
    
    if prediction == -1:
        return jsonify({"error": "Match not found"})
    
    explanation_list = exp.as_list()
    explanation_html = """
    <div style='text-align: center; margin-top: 10px;'>
    <table style='font-family: monospace; text-align: left; display: inline-block;'>
    """ + "\n".join([
        f"<tr>"
        f"<td>{feat}</td>"
        f"<td style='padding: 0 2em;'>→</td>"
        f"<td style='color:{'limegreen' if weight > 0 else 'tomato'};'>"
        f"{'+' if weight > 0 else ''}{weight * 100:.2f}%"
        f"</td>"
        f"</tr>"
        for feat, weight in explanation_list
    ]) + """
    </table>
    </div>
    """

    # Return the probability for blue team winning (a float between 0 and 1)
    return jsonify({
         "probability": prediction,
         "explanation": explanation_html
    })


@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, port=8080)