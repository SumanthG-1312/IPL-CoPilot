import pandas as pd

# Path to dataset
DATA_PATH = "data/ipl_ball_by_ball.csv"

# Load dataset
df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset loaded successfully!")

print("\nOriginal shape:")
print(df.shape)


# -----------------------------
# Data Cleaning
# -----------------------------

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

# Convert string columns to clean strings
string_columns = [
    "batter",
    "bowler",
    "non-striker",
    "BattingTeam"
]

for column in string_columns:
    df[column] = df[column].astype(str).str.strip()


# Fill missing values in event-related columns
df["extra_type"] = df["extra_type"].fillna("None")
df["player_out"] = df["player_out"].fillna("None")
df["kind"] = df["kind"].fillna("None")
df["fielders_involved"] = df["fielders_involved"].fillna("None")


# Remove duplicate rows
df = df.drop_duplicates()


# -----------------------------
# Final information
# -----------------------------

print("\nCleaned shape:")
print(df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nSample data:")
print(df.head())


# Save cleaned dataset
OUTPUT_PATH = "data/ipl_ball_by_ball_cleaned.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully!")
print(f"File: {OUTPUT_PATH}")