import pandas as pd

# Load the dataset
df = pd.read_csv("train.csv")

print("=== Data Sanity Check ===")
print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

# Winner stats
total_winners = df["winner"].sum()
print(f"Total winners: {total_winners} ({total_winners / len(df) * 100:.2f}%)")
print()

# Winners by branch
print("Winners by branch:")
branch_winners = df[df["winner"] == 1]["branch"].value_counts()
print(branch_winners)
print()

# Winners by experience
print("Winners by experience:")
exp_winners = df[df["winner"] == 1]["experience"].value_counts()
print(exp_winners)
print()

# Winners by event_type
print("Winners by event_type:")
event_winners = df[df["winner"] == 1]["event_type"].value_counts()
print(event_winners)
print()

# Null counts
print("Null counts:")
nulls = df.isnull().sum()
print(nulls[nulls > 0])
print()

# Feature distributions
print("Feature distributions (means):")
numerical_cols = [
    "leetcode_hours",
    "leetcode_problems",
    "team_size",
    "gaming_hours",
    "instagram_hours",
    "social_skill_points",
    "connections_among_seniors",
    "connections_with_faculty",
    "connections_among_juniors",
    "gender_ratio",
]
for col in numerical_cols:
    if col in df.columns:
        mean_val = df[col].mean()
        print(f"{col}: {mean_val:.2f}")
print()

# Outlier check (simple: max values)
print("Max values (potential outliers):")
for col in numerical_cols:
    if col in df.columns:
        max_val = df[col].max()
        print(f"{col}: {max_val}")
print()

# Categorical distributions
print("Branch distribution:")
print(df["branch"].value_counts())
print()

print("Experience distribution:")
print(df["experience"].value_counts())
print()

print("Event type distribution:")
print(df["event_type"].value_counts())
print()

print("Sanity check complete.")
