import pandas as pd
from ydata_profiling import ProfileReport

# Load dataset
column_names = [
    "matchId",
    "blueTeamControlWardsPlaced", "blueTeamWardsPlaced", "blueTeamTotalKills",
    "blueTeamDragonKills", "blueTeamHeraldKills", "blueTeamTowersDestroyed",
    "blueTeamInhibitorsDestroyed", "blueTeamTurretPlatesDestroyed",
    "blueTeamFirstBlood", "blueTeamMinionsKilled", "blueTeamJungleMinions",
    "blueTeamTotalGold", "blueTeamXp", "blueTeamTotalDamageToChamps",
    "redTeamControlWardsPlaced", "redTeamWardsPlaced", "redTeamTotalKills",
    "redTeamDragonKills", "redTeamHeraldKills", "redTeamTowersDestroyed",
    "redTeamInhibitorsDestroyed", "redTeamTurretPlatesDestroyed",
    "redTeamMinionsKilled", "redTeamJungleMinions", "redTeamTotalGold",
    "redTeamXp", "redTeamTotalDamageToChamps",
    "blueWin", "end"
]

df = pd.read_csv("match_data_v5.csv", names=column_names, header=0, index_col=0)

print(df.head()) # Display the first 5 rows of the dataset

df = df.iloc[:,:-1] # Drop the last column

print(df.head()) 

# Check for duplicates
num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")

duplicates = df[df.duplicated()]
print(duplicates)

df_cleaned = df.drop_duplicates() # Drop duplicates

num_duplicates = df_cleaned.duplicated().sum()
print(f"Number of duplicate rows after cleanup: {num_duplicates}")

# Generate report
profile = ProfileReport(df_cleaned, title="EDA Report", explorative=True)
profile.to_file("eda_report.html")
