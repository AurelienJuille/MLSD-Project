from kfp.dsl import (
    Dataset,     # For handling datasets
    Input,       # For component inputs
    Model,       # For handling ML models
    Output,      # For component outputs
    Metrics,     # For tracking metrics
    component,   # For creating pipeline components
)

@component(
    base_image=f"europe-west1-docker.pkg.dev/lolffate/lolffate-pipeline/training:latest",
    output_component_file="training.yaml"
)
def training(
    preprocessed_dataset_train: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
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
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV, KFold

    df = pd.read_csv(preprocessed_dataset_train.path)
    
    # Split features and target
    X_train = df.drop(columns=["blueWin"]) # Features
    y_train = df["blueWin"] # Labels
    
    # Define hyperparameters to tune
    param_grid = {
        'model__n_estimators': [200, 300, 500],
        'model__criterion': ['entropy'],
        'model__max_depth': [None, 3, 4, 5, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__bootstrap': [True, False]
    }
    
    # Define scaler and model
    pipeline = Pipeline([
        ("scaler", preprocessing.StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=300,
                                        criterion='entropy',
                                        max_depth=10,
                                        min_samples_split=5,
                                        min_samples_leaf=2,
                                        n_jobs=-1,
                                        random_state=0))
    ])
    
    # Define cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    # Search for best hyperparameters using RandomizedSearchCV
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, scoring='neg_log_loss', n_jobs=-1, cv=cv)
    random_search.fit(X_train, y_train)

    # Get best estimator
    pipeline = random_search.best_estimator_
    
    # Create the model directory if it doesn't exist
    os.makedirs(model.path, exist_ok=True)
    
    # Save the model
    joblib.dump(pipeline, os.path.join(model.path, "model.joblib"))
    
    logging.info(f"Model saved to: {model.path}")
    
    # Get best training score
    best_score = random_search.best_score_
    metrics.log_metric("best_score", best_score)
    
    # Get best param
    param_sample = random_search.best_params_
    metrics.log_metric("best_param", param_sample)

    
