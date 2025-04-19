import pandas as pd
import lime
import lime.lime_tabular
import joblib
import tempfile

from flask import Flask, request, render_template, jsonify, redirect, url_for
from google.cloud import aiplatform
from google.cloud import storage

app = Flask(__name__)

history = []

aiplatform.init(project="lolffate", location="europe-west1") # Initialize the AI Platform project and location

models = aiplatform.Model.list(filter='display_name="lolffate"') # Get the list of models with the display name "lolffate"
latest_model = sorted(models, key=lambda m: m.create_time, reverse=True)[0] # Get the latest model

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

model = load_artifact_from_gcs(latest_model.uri, "model.joblib") # Load the model from GCS once at startup

gcs_path = 'gs://lolffate-data/data/processed_dataset.csv' # Path to the dataset in GCS

def create_global_explainer():
    # Load the dataset
    df_rep = pd.read_csv(gcs_path, index_col=0).drop(columns=["blueWin"])
    
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
    to_predict = features.values.reshape(1, -1)

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
    df = pd.read_csv(gcs_path, index_col=0)
    df = df.drop(columns=["blueWin"])
    
    print(df.columns)
    
    try:
        match_features = df.loc[match_id]
    except KeyError:
        return -1, None # Match not found
    
    prediction = predict(match_features)
    
    history.append({
        "match_id": match_id,
        "prediction": prediction
    })

    # Get the explanation for the prediction
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