import pandas as pd

# Load the dataset
df = pd.read_csv("regression/train.csv")

print("=== Regression Problem Sanity Check ===")
print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

# Target stats
total_participants = df["participant_count"].sum()
print(f"Total participants across all events: {total_participants}")
print(f"Average participants per event: {df['participant_count'].mean():.2f}")
print(f"Min participants: {df['participant_count'].min()}")
print(f"Max participants: {df['participant_count'].max()}")
print()

# Participants by event_type
print("Average participants by event_type:")
avg_by_event = (
    df.groupby("event_type")["participant_count"].mean().sort_values(ascending=False)
)
print(avg_by_event)
print()

# Participants by guest
print("Average participants by guest:")
avg_by_guest = (
    df.groupby("guest")["participant_count"].mean().sort_values(ascending=False)
)
print(avg_by_guest)
print()

# Null counts
print("Null counts:")
nulls = df.isnull().sum()
print(nulls[nulls > 0])
print()

# Feature distributions
print("Feature distributions (means):")
numerical_cols = [
    "promotion_level",
    "event_duration",
    "venue_capacity",
    "registration_fee",
    "social_media_buzz",
    "concurrent_events_count",
]
for col in numerical_cols:
    if col in df.columns:
        mean_val = df[col].mean()
        print(f"{col}: {mean_val:.2f}")
print()

# Categorical distributions
print("Event type distribution:")
print(df["event_type"].value_counts())
print()

print("Guest distribution:")
print(df["guest"].value_counts())
print()

print("Organising department distribution:")
print(df["organising_department"].value_counts())
print()

print("Sanity check complete.")
