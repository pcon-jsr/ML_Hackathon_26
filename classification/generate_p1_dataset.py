'''                                 _       _       
     _ __   ___  _ __ ___ ___ _   _| | __ _(_)_ __  
    | '_ \ / _ \| '__/ __/ _ \ | | | |/ _` | | '_ \ 
    | |_) | (_) | | | (_|  __/ |_| | | (_| | | | | |
    | .__/ \___/|_|  \___\___|\__, |_|\__,_|_|_| |_|
    |_|                       |___/                 

'''
import numpy as np
import pandas as pd
import random

SEED = 69
np.random.seed(SEED)
random.seed(SEED)

# Dataset sizes
N_TRAIN = 50000
N_EVAL = 10000
TOTAL = N_TRAIN + N_EVAL

# Categorical feature weights (must sum to 1.0)
EXPERIENCE_LEVELS = ["beginner", "intermediate", "expert"]
EXPERIENCE_WEIGHTS = [0.5, 0.3, 0.2]  # Beginners most common

BRANCHES = ["CSE", "ECE", "EE", "CE", "MM", "ME", "ECM", "PIE"]
BRANCH_WEIGHTS = [0.25, 0.2, 0.1, 0.08, 0.08, 0.12, 0.12, 0.05]  # CSE most common

EVENT_TYPES = [
    "hackathon",
    "contest",
    "sporting",
    "gaming",
    "dance",
    "singing",
    "fun",
    "debate",
    "other",
]
EVENT_WEIGHTS = [
    0.3,
    0.2,
    0.1,
    0.1,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
]  # Hackathon most common

# Branch scores (higher = better baseline performance)
BRANCH_SCORES = {
    "CSE": 4.0,
    "ECE": 2.5,
    "EE": 2.0,
    "ME": 3.0,
    "ECM": 3.0,
    "CE": 1.0,
    "MM": 1.0,
    "PIE": 0.5,
}

# Event-type and branch interaction scores (added to base score)
EVENT_BRANCH_SCORES = {
    "hackathon": {"CSE": 3.0, "ECE": 2.0, "default": 1.0},  # CSE best, ECE good
    "contest": {"CSE": 3.0, "ECE": 2.0, "default": 1.0},  # Similar to hackathon
    "sporting": {"CSE": -1.0, "ECE": 1.0, "default": 0.5},  # CSE weak, ECE okay
    "gaming": {"CSE": 1.5, "ECE": 1.0, "default": 0.5},
    "dance": {"CSE": 0.0, "ECE": 0.0, "default": 1.0},  # Neutral, favor others
    "singing": {"CSE": 0.0, "ECE": 0.0, "default": 1.0},
    "fun": {"CSE": 0.5, "ECE": 0.5, "default": 1.0},
    "debate": {"CSE": 1.0, "ECE": 1.0, "default": 1.0},
    "other": {"CSE": 0.5, "ECE": 0.5, "default": 0.5},
}

# Winner score formula weights (adjust for feature importance)
SCORE_WEIGHTS = {
    "leetcode_hours_log": 0.1,
    "leetcode_problems": 0.04,
    "team_size": 0.075,
    "gaming_hours": -0.125,
    "instagram_hours": -0.06,
    "social_skill_points": 0.09,
    "connections_seniors": 0.2,
    "connections_faculty": 0.175,
    "connections_juniors": 0.02,
    "have_freshman": -0.1,
    "all_freshman": -0.2,
    "experience_score": 0.125,
    "branch_score": 0.15,
    "event_branch_score": 0.1,
    "interaction_leetcode_exp": 0.025,
    "interaction_social_team": 0.01,
}

# Score bias and noise
SCORE_BIAS = -15.0  # Negative bias for ~20-25% winners
SCORE_NOISE_STD = 1.5  # Normal noise std

# Outlier probabilities and effects
OUTLIER_PROB = 0.069  # 6.9% chance for outliers
OUTLIER_FEATURES = ["leetcode_hours", "gaming_hours", "connections_among_seniors"]

# Null probabilities (random ranges for realism)
NULL_PROBS_BASE = 0.02
NULL_PROBS_FEATURES = {
    "leetcode_hours": (0.03, 0.07),
    "team_size": (0.01, 0.03),
    "gaming_hours": (0.015, 0.025),
    "instagram_hours": (0.02, 0.04),
    "social_skill_points": (0.01, 0.02),
    "connections_among_seniors": (0.025, 0.035),
    "connections_with_faculty": (0.015, 0.025),
    "connections_among_juniors": (0.02, 0.03),
    "have_freshman": (0.005, 0.015),
    "all_freshman": (0.005, 0.015),
}

# Experience score mapping
EXPERIENCE_SCORES = {"beginner": 0, "intermediate": 1, "expert": 2}

# ========================================
# DATA GENERATION
# ========================================

data = []

for i in range(TOTAL):
    # Categorical selections
    experience = random.choices(EXPERIENCE_LEVELS, weights=EXPERIENCE_WEIGHTS, k=1)[0]
    branch = random.choices(BRANCHES, weights=BRANCH_WEIGHTS, k=1)[0]
    event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS, k=1)[0]

    # Numerical features
    leetcode_hours = (
        int(np.random.exponential(3)) % 11
        if branch in ["CSE", "ECE", "EE"]
        else int(np.random.exponential(1.5)) % 11
    )
    leetcode_problems = int(np.random.normal(50, 20))
    team_size = np.random.randint(1, 6)
    gaming_hours = 0 if random.random() < 0.3 else np.random.randint(5, 9)
    instagram_hours = 0 if random.random() < 0.4 else np.random.randint(5, 9)
    social_skill_points = int(np.random.beta(2, 2) * 100)
    connections_among_seniors = np.random.randint(0, 21)
    connections_with_faculty = np.random.randint(0, 11)
    connections_among_juniors = np.random.randint(0, 31)
    have_freshman = random.choice([0, 1])
    all_freshman = random.choice([0, 1]) if have_freshman == 1 else 0

    # Gender ratio: fraction of females
    num_females = random.randint(0, team_size)
    gender_ratio = round(num_females / team_size, 2)

    # Scores
    exp_score = EXPERIENCE_SCORES[experience]
    branch_score = BRANCH_SCORES[branch]
    event_branch_score = EVENT_BRANCH_SCORES[event_type].get(
        branch, EVENT_BRANCH_SCORES[event_type]["default"]
    )

    # Winner score formula
    base_score = (
        np.log(1 + leetcode_hours) * SCORE_WEIGHTS["leetcode_hours_log"]
        + leetcode_problems * SCORE_WEIGHTS["leetcode_problems"]
        + team_size * SCORE_WEIGHTS["team_size"]
        + gaming_hours * SCORE_WEIGHTS["gaming_hours"]
        + instagram_hours * SCORE_WEIGHTS["instagram_hours"]
        + social_skill_points * SCORE_WEIGHTS["social_skill_points"]
        + connections_among_seniors * SCORE_WEIGHTS["connections_seniors"]
        + connections_with_faculty * SCORE_WEIGHTS["connections_faculty"]
        + connections_among_juniors * SCORE_WEIGHTS["connections_juniors"]
        + have_freshman * SCORE_WEIGHTS["have_freshman"]
        + all_freshman * SCORE_WEIGHTS["all_freshman"]
        + exp_score * SCORE_WEIGHTS["experience_score"]
        + branch_score * SCORE_WEIGHTS["branch_score"]
        + event_branch_score * SCORE_WEIGHTS["event_branch_score"]
        + leetcode_hours * exp_score * SCORE_WEIGHTS["interaction_leetcode_exp"]
        + social_skill_points * team_size * SCORE_WEIGHTS["interaction_social_team"]
    )

    score = base_score + SCORE_BIAS + np.random.normal(0, SCORE_NOISE_STD)
    prob = 1 / (1 + np.exp(-score))
    winner = 1 if prob > 0.5 else 0
    if random.random() < 0.2:  # 20% random
        winner = random.choice([0, 1])

    # Outliers
    if random.random() < OUTLIER_PROB:
        outlier_feat = random.choice(OUTLIER_FEATURES)
        if outlier_feat == "leetcode_hours":
            leetcode_hours = np.random.choice([0, 20])
        elif outlier_feat == "gaming_hours":
            gaming_hours = int(np.random.uniform(10, 20))
        elif outlier_feat == "connections_among_seniors":
            connections_among_seniors = np.random.randint(50, 100)

    # Nulls
    if random.random() < random.uniform(
        *NULL_PROBS_FEATURES.get("leetcode_hours", (0.03, 0.07))
    ):
        leetcode_hours = np.nan
        leetcode_problems = np.nan
    for field, (low, high) in NULL_PROBS_FEATURES.items():
        if field not in ["leetcode_hours"] and random.random() < random.uniform(
            low, high
        ):
            locals()[field] = np.nan

    data.append(
        {
            "id": str(i + 1).zfill(3),
            "leetcode_hours": max(0, leetcode_hours)
            if not np.isnan(leetcode_hours)
            else np.nan,
            "leetcode_problems": max(0, leetcode_problems)
            if not np.isnan(leetcode_problems)
            else np.nan,
            "team_size": team_size,
            "experience": experience,
            "branch": branch,
            "gaming_hours": max(0, gaming_hours)
            if not np.isnan(gaming_hours)
            else np.nan,
            "instagram_hours": max(0, instagram_hours)
            if not np.isnan(instagram_hours)
            else np.nan,
            "social_skill_points": np.clip(social_skill_points, 0, 100)
            if not np.isnan(social_skill_points)
            else np.nan,
            "connections_among_seniors": connections_among_seniors,
            "connections_with_faculty": connections_with_faculty,
            "connections_among_juniors": connections_among_juniors,
            "have_freshman": have_freshman,
            "all_freshman": all_freshman,
            "gender_ratio": gender_ratio,
            "event_type": event_type,
            "winner": winner,
        }
    )

# ========================================
# SAVE DATASETS
# ========================================

df = pd.DataFrame(data)
train_df = df[:N_TRAIN]
eval_df = df[N_TRAIN:]

train_df.to_csv("train.csv", index=False)
eval_df.drop("winner", axis=1).to_csv("eval.csv", index=False)
eval_df[["winner"]].to_csv("eval_labels.csv", index=False)

print(
    f"Datasets generated: train.csv ({N_TRAIN}), eval.csv ({N_EVAL} features), eval_labels.csv ({N_EVAL} labels)"
)
