from kfp.dsl import (
    Dataset,     # For handling datasets
    Input,       # For component inputs
    Model,       # For handling ML models
    Output,      # For component outputs
    Metrics,     # For tracking metrics
    HTML,        # For visualization
    component,   # For creating pipeline components
)

@component(
    base_image=f"europe-west1-docker.pkg.dev/lolffate/lolffate-pipeline/training:latest",
    output_component_file="evaluation.yaml"
)
def evaluation(
    preprocessed_dataset_test: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics]
):
    """
    Evaluates the model's performance and generates visualizations.
    
    Args:
        model: Input trained model
        preprocessed_dataset: Input preprocessed dataset
        metrics: Output artifact for evaluation metrics
        html: Output artifact for visualization HTML
    """
    import pandas as pd
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score
    import logging

    # Load the model and dataset
    model = joblib.load(model.path + "/model.joblib")
    
    # Load the test dataset
    df = pd.read_csv(preprocessed_dataset_test.path)
    X = df.drop(columns=["blueWin"])
    y = df["blueWin"]
    
    # Make predictions
    y_pred = model.predict(X)

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    accuracy = model.score(X, y)
    
    # Save the metrics
    metrics.log_metric("mean_squared_error", mse)
    metrics.log_metric("r2_score", r2)
    metrics.log_metric("accuracy", accuracy)
    
    logging.info(f"Evaluation metrics: MSE={mse}, R2={r2}, Accuracy={accuracy}")