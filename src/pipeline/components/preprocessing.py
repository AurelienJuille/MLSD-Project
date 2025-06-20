from kfp.dsl import Dataset  # For handling datasets
from kfp.dsl import Input  # For component inputs
from kfp.dsl import Output  # For component outputs
from kfp.dsl import component  # For creating pipeline components


@component(
    base_image=f"europe-west1-docker.pkg.dev/lolffate/lolffate-pipeline/training:latest",
    output_component_file="preprocessing.yaml",
)
def preprocessing(
    dataset: Input[Dataset],
    preprocessed_dataset: Output[Dataset],
):
    """
    Preprocesses the dataset for training.

    Args:
        input_dataset: Input dataset from the data ingestion step
        preprocessed_dataset: Output artifact for the preprocessed dataset
    """
    import logging

    import pandas as pd
    from google.cloud import storage

    def upload_blob(bucket_name, source_file_path, destination_blob_name):
        """Uploads a file to the specified bucket."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_path)

        print(f"File {source_file_path} uploaded to {destination_blob_name}.")

    # Load the dataset
    df = pd.read_csv(dataset.path, index_col=0)

    # Drop unnecessary columns
    df = df.drop(columns=["end"])

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Add new features
    df["diffMinionsKilled"] = df["blueTeamMinionsKilled"] - df["redTeamMinionsKilled"]
    df["diffJungleMinions"] = df["blueTeamJungleMinions"] - df["redTeamJungleMinions"]
    df["diffTotalGold"] = df["blueTeamTotalGold"] - df["redTeamTotalGold"]
    df["diffTotalKills"] = df["blueTeamTotalKills"] - df["redTeamTotalKills"]
    df["diffXp"] = df["blueTeamXp"] - df["redTeamXp"]
    df["diffTotalDamageToChamps"] = (
        df["blueTeamTotalDamageToChamps"] - df["redTeamTotalDamageToChamps"]
    )
    df["diffDragonKills"] = df["blueTeamDragonKills"] - df["redTeamDragonKills"]
    df["diffHeraldKills"] = df["blueTeamHeraldKills"] - df["redTeamHeraldKills"]
    df["diffTowersDestroyed"] = (
        df["blueTeamTowersDestroyed"] - df["redTeamTowersDestroyed"]
    )
    df["diffInhibitorsDestroyed"] = (
        df["blueTeamInhibitorsDestroyed"] - df["redTeamInhibitorsDestroyed"]
    )
    df["diffTurretPlatesDestroyed"] = (
        df["blueTeamTurretPlatesDestroyed"] - df["redTeamTurretPlatesDestroyed"]
    )

    # Save preprocessed dataset
    df.to_csv(preprocessed_dataset.path, index=True)
    logging.info(f"Preprocessed dataset saved to: {preprocessed_dataset.path}.")

    # Upload preprocessed dataset to GCS bucket
    upload_blob(
        "lolffate-data", preprocessed_dataset.path, "data/processed_dataset.csv"
    )
