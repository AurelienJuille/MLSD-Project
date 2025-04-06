from kfp.dsl import (
    Dataset,     # For handling datasets
    Input,       # For component inputs
    Model,       # For handling ML models
    Output,      # For component outputs
    component,   # For creating pipeline components
)

@component(
    base_image=f"europe-west1-docker.pkg.dev/lolffate/lolffate-pipeline/training:latest",
    output_component_file="training.yaml"
)
def training(
    preprocessed_dataset_train: Input[Dataset],
    model: Output[Model]
):
    """
    Trains the model on the preprocessed dataset.
    
    Args:
        preprocessed_dataset: Input preprocessed dataset
        model: Output artifact for the trained model
        metrics: Output artifact for training metrics
        hyperparameters: Dictionary of hyperparameters
    """
    import os
    import pandas as pd
    import joblib
    import logging
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    

    df = pd.read_csv(preprocessed_dataset_train.path)
    
    # Split features and target
    X_train = df.drop(columns=["blueWin"]) # Features
    y_train = df["blueWin"] # Labels
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    
    # Initialize and train the model
    rf_model = RandomForestClassifier(n_estimators=300,
                                   min_samples_split=10,
                                   min_samples_leaf=4,
                                   max_depth=10,
                                   criterion='entropy',
                                   bootstrap=True)
    
    rf_model.fit(X_train, y_train)
    
    # Create the model directory if it doesn't exist
    os.makedirs(model.path, exist_ok=True)
    
    # Save the model
    joblib.dump(rf_model, os.path.join(model.path, "model.joblib"))
    
    logging.info(f"Model saved to: {model.path}")
