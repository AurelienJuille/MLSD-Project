from kfp.dsl import (
    Dataset,  # For handling datasets
    Input,  # For component inputs
    Output,  # For component outputs
    component,  # For creating pipeline components
)


@component(
    base_image=f"europe-west1-docker.pkg.dev/lolffate/lolffate-pipeline/training:latest",
    output_component_file="train_test_split.yaml",
)
def train_test_split(
    preprocessed_dataset: Input[Dataset],
    preprocessed_dataset_train: Output[Dataset],
    preprocessed_dataset_test: Output[Dataset],
):
    """
    Splits the preprocessed dataset into training and testing sets.

    Args:
        preprocessed_dataset: Input preprocessed dataset
        preprocessed_dataset_train: Output artifact for the training dataset
        preprocessed_dataset_test: Output artifact for the testing dataset
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(preprocessed_dataset.path)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    df_train.to_csv(preprocessed_dataset_train.path, index=False)
    df_test.to_csv(preprocessed_dataset_test.path, index=False)
