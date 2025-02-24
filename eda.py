import kagglehub

# Download latest version
path = kagglehub.dataset_download("karlorusovan/league-of-legends-soloq-matches-at-10-minutes-2024")

print("Path to dataset files:", path)