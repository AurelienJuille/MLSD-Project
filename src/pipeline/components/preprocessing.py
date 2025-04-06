from kfp.dsl import (
    Dataset,     # For handling datasets
    Input,       # For component inputs
    Output,      # For component outputs
    component,   # For creating pipeline components
)

@component(
    base_image=f"europe-west1-docker.pkg.dev/lolffate/lolffate-pipeline/training:latest",
    output_component_file="preprocessing.yaml"
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
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    import logging
    
    # Load the dataset
    df = pd.read_csv(dataset.path)

    # Drop unnecessary columns
    df = df.drop(columns=["end"])
    
    # Drop duplicate rows
    df = df.drop_duplicates()
    
    # Add new features
    df['diffMinionsKilled'] = (df['blueTeamMinionsKilled'] - df['redTeamMinionsKilled'])
    df['diffJungleMinions'] = (df['blueTeamJungleMinions'] - df['redTeamJungleMinions'])
    df['diffTotalGold'] = (df['blueTeamTotalGold'] - df['redTeamTotalGold'])
    df['diffTotalKills'] = (df['blueTeamTotalKills'] - df['redTeamTotalKills'])
    df['diffXp'] = (df['blueTeamXp'] - df['redTeamXp'])
    df['diffTotalDamageToChamps'] = (df['blueTeamTotalDamageToChamps'] - df['redTeamTotalDamageToChamps'])
    df['diffDragonKills'] = (df['blueTeamDragonKills'] - df['redTeamDragonKills'])
    df['diffHeraldKills'] = (df['blueTeamHeraldKills'] - df['redTeamHeraldKills'])
    df['diffTowersDestroyed'] = (df['blueTeamTowersDestroyed'] - df['redTeamTowersDestroyed'])
    df['diffInhibitorsDestroyed'] = (df['blueTeamInhibitorsDestroyed'] - df['redTeamInhibitorsDestroyed'])
    df['diffTurretPlatesDestroyed'] = (df['blueTeamTurretPlatesDestroyed'] - df['redTeamTurretPlatesDestroyed'])

    # Save preprocessed dataset
    df.to_csv(preprocessed_dataset.path, index=False)
    logging.info(f"Preprocessed dataset saved to: {preprocessed_dataset.path}.")