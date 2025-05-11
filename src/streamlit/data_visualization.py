import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import joblib
import lime
import lime.lime_tabular
from urllib.request import Request, urlopen
import streamlit.components.v1 as components

url_model = "https://storage.googleapis.com/lolffate-data/data/model.joblib"
req_model = Request(url_model, headers={'User-Agent': 'Mozilla/5.0'})
model = joblib.load(urlopen(req_model))

url_processed = 'https://storage.googleapis.com/lolffate-data/data/processed_dataset.csv'
req_processed = Request(url_processed, headers={'User-Agent': 'Mozilla/5.0'})
match_processed = pd.read_csv(urlopen(req_processed), index_col=0)

def create_global_explainer():
    """
    Create a global LIME explainer for the model.
    
    Returns:
        explainer: LIME explainer object
    """
    df_rep = match_processed.drop(columns=["blueWin"]) # Drop the target variable

    X_train = df_rep.apply(lambda row: row, axis=1) 
    X_train = X_train.to_numpy() 
    
    feature_names = list(df_rep.iloc[0].index) # Extract feature names

    explainer = lime.lime_tabular.LimeTabularExplainer( # Create the explainer
        training_data=X_train,
        feature_names=feature_names,
        class_names=["Red team", "Blue team"],
        mode="classification",
    ) 
    
    return explainer

# Create the global explainer once, at startup
global_explainer = create_global_explainer()

def predict_match(match_id):
    """
    Predict the result of a match

    Args:
        match_id: id of the match to predict

    Returns:
        prediction: prediction result
        (probability of blue team winning, float [0, 1])
        or -1 if match not found
        exp: explanation of the prediction using LIME
    """
    df = match_processed.drop(columns=["blueWin"])

    try:
        match_features = df.loc[match_id]
    except KeyError:
        return -1, None  # Match not found

    to_predict = match_features.values.reshape(1, -1)
    prediction = model.predict_proba(to_predict)[0, 1] # Probability of blue team winning for the first sample

    exp = global_explainer.explain_instance( # Get the explanation for the prediction
        match_features.values, model.predict_proba, num_features=5
    )

    return prediction, exp

# Sidebar
visualization = st.sidebar.selectbox("Select visualization type", ["Predictions", "Dataset"])

# Main panel
if visualization == "Predictions":
    st.title("Predictions")
    
    with st.form(key="prediction_form"):
        st.subheader("Enter Game ID")
        game_id = st.text_input("Game ID")
        if st.form_submit_button("Predict"):
            if game_id:
                prediction, exp = predict_match(game_id)
                
                if prediction == -1:
                    st.warning("The game ID you entered does not exist in the dataset.")
                
                else:
                    st.subheader("Prediction:")
                    st.write(f"Probability of blue team winning: {prediction:.2f}")
                    
                    st.write("Prediction explanation:")
                    st.write("The following features contributed to the prediction:")
                    
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
                    components.html(explanation_html)
                    
            else:
                st.warning("Please enter a Game ID.")
                
    id_example_list = ["EUW1_6882489515'", "EUW1_6882416210'", "EUW1_6881092720'", "EUW1_6879405717'", "EUW1_6879389461'"]
    st.subheader("Example Game IDs")
    st.write("Here are some example Game IDs you can use for testing:")
    st.write(id_example_list)
    
elif visualization == "Dataset":
    st.title("Processed Data Visualization")
    
    st.subheader("Processed Data Preview")
    st.write(match_processed.head())
    
    st.subheader("Processed Data Statistics")
    st.write(match_processed.describe())

    st.subheader("Feature Distribution")
    feature = st.selectbox("Select feature to visualize", match_processed.columns)
    st.write(f"Distribution of {feature}:")
    
    fig, ax = plt.subplots()
    match_processed[feature].hist(bins=30, ax=ax)
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {feature}")
    st.pyplot(fig)
    
    st.subheader("Correlation Heatmap of selected features")
    selected_features = st.multiselect("Select features to visualize", match_processed.columns, default=match_processed.columns[:5])
    if selected_features:
        fig, ax = plt.subplots()
        
        corr = match_processed[selected_features].corr()
        cax = ax.matshow(corr, cmap="coolwarm")
        plt.colorbar(cax)
        
        ax.set_xticks(range(len(selected_features)))
        ax.set_yticks(range(len(selected_features)))
        ax.set_xticklabels(selected_features, rotation=90)
        ax.set_yticklabels(selected_features)
        ax.set_title("Correlation heatmap of selected features")
        
        st.pyplot(fig)
    else:
        st.write("Please select features to visualize the correlation heatmap.")
    
