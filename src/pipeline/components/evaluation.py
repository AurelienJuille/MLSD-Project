from kfp.dsl import HTML  # For visualization
from kfp.dsl import Dataset  # For handling datasets
from kfp.dsl import Input  # For component inputs
from kfp.dsl import Metrics  # For tracking metrics
from kfp.dsl import Model  # For handling ML models
from kfp.dsl import Output  # For component outputs
from kfp.dsl import component  # For creating pipeline components


@component(
    base_image=f"europe-west1-docker.pkg.dev/lolffate/lolffate-pipeline/training:latest",
    output_component_file="evaluation.yaml",
)
def evaluation(
    preprocessed_dataset_test: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics],
):
    """
    Evaluates the model's performance and generates visualizations.

    Args:
        model: Input trained model
        preprocessed_dataset: Input preprocessed dataset
        metrics: Output artifact for evaluation metrics
        html: Output artifact for visualization HTML
    """
    import logging

    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score

    # Load the model and dataset
    model = joblib.load(model.path + "/model.joblib")

    # Load the test dataset
    df = pd.read_csv(preprocessed_dataset_test.path, index_col=0)
    X = df.drop(columns=["blueWin"])
    y = df["blueWin"]

    # Make predictions
    y_pred = model.predict(X)

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Save the metrics
    metrics.log_metric("mean_squared_error", mse)
    metrics.log_metric("r2_score", r2)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)

    # Save the accuracy metric
    metrics.log_metric("accuracy", accuracy)

    # Make predictions for log loss
    y_pred_prob = model.predict_proba(X)[:, 1]

    # Calculate log loss
    loss = log_loss(y, y_pred_prob)

    # Save the log loss metric
    metrics.log_metric("log_loss", loss)

    logging.info(
        f"Evaluation metrics: MSE={mse}, R2={r2}, Accuracy={accuracy}, Log Loss={loss}"
    )
