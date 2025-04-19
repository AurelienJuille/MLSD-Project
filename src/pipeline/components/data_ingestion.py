from kfp.dsl import (
    Dataset,  # For handling datasets
    Output,  # For component outputs
    component,  # For creating pipeline components
)


@component(
    base_image=f"europe-west1-docker.pkg.dev/lolffate/lolffate-pipeline/training:latest",
    output_component_file="data_ingestion.yaml",
)
def data_ingestion(dataset: Output[Dataset]):
    """
    Loads and prepares the house price dataset.

    Args:
        dataset: Output artifact to store the prepared dataset
    """
    import pandas as pd
    import logging

    try:
        logging.info("Starting data ingestion...")

        # Load the dataset
        column_names = [
            "matchId",
            "blueTeamControlWardsPlaced",
            "blueTeamWardsPlaced",
            "blueTeamTotalKills",
            "blueTeamDragonKills",
            "blueTeamHeraldKills",
            "blueTeamTowersDestroyed",
            "blueTeamInhibitorsDestroyed",
            "blueTeamTurretPlatesDestroyed",
            "blueTeamFirstBlood",
            "blueTeamMinionsKilled",
            "blueTeamJungleMinions",
            "blueTeamTotalGold",
            "blueTeamXp",
            "blueTeamTotalDamageToChamps",
            "redTeamControlWardsPlaced",
            "redTeamWardsPlaced",
            "redTeamTotalKills",
            "redTeamDragonKills",
            "redTeamHeraldKills",
            "redTeamTowersDestroyed",
            "redTeamInhibitorsDestroyed",
            "redTeamTurretPlatesDestroyed",
            "redTeamMinionsKilled",
            "redTeamJungleMinions",
            "redTeamTotalGold",
            "redTeamXp",
            "redTeamTotalDamageToChamps",
            "blueWin",
            "end",
        ]

        df = pd.read_csv(
            "gs://lolffate-data/data/match_data_v5.csv",
            names=column_names,
            header=0,
            index_col=0,
        )
        logging.info("Dataset loaded successfully.")

        # Save the dataset
        logging.info(f"Saving dataset to {dataset.path}...")
        df.to_csv(dataset.path, index=True)
        logging.info(f"Dataset saved to: {dataset.path}.")

    except Exception as e:
        logging.error(f"Error during data ingestion: {e}")
