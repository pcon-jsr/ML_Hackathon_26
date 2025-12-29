import numpy as np
import pandas as pd
import random

# ========================================
# CONFIGURABLE KNOBS AND WEIGHTS
# ========================================

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Dataset sizes
N_TRAIN = 50000
N_EVAL = 10000
TOTAL = N_TRAIN + N_EVAL

# Categorical feature weights (must sum to 1.0)
EVENT_TYPES = [
    "hackathon",
    "workshop",
    "seminar",
    "contest",
    "fun",
    "sport",
    "talk",
    "presentation",
    "gaming",
    "social service",
    "educational",
    "others",
]
EVENT_WEIGHTS = [
    0.15,
    0.1,
    0.1,
    0.08,
    0.08,
    0.08,
    0.08,
    0.08,
    0.06,
    0.06,
    0.06,
    0.07,
]  # Hackathon most common

GUESTS = [
    "political",
    "youtuber",
    "professional",
    "influencer",
    "official",
    "alumni",
    "faculty",
    "industrialist",
]
# Guest weights per event_type for realism (e.g., politicians for talks, youtubers for fun events)
GUEST_WEIGHTS_PER_EVENT = {
    "hackathon": [
        0.05,
        0.2,
        0.2,
        0.15,
        0.05,
        0.1,
        0.1,
        0.15,
    ],  # YouTuber, professional, influencer high
    "workshop": [
        0.05,
        0.1,
        0.25,
        0.1,
        0.05,
        0.15,
        0.15,
        0.1,
    ],  # Professional, alumni, faculty high
    "seminar": [
        0.15,
        0.05,
        0.15,
        0.05,
        0.15,
        0.1,
        0.2,
        0.1,
    ],  # Political, professional, faculty high
    "contest": [
        0.05,
        0.15,
        0.15,
        0.2,
        0.05,
        0.1,
        0.1,
        0.15,
    ],  # Influencer, youtuber high
    "fun": [0.1, 0.2, 0.1, 0.2, 0.05, 0.1, 0.05, 0.1],  # YouTuber, influencer high
    "sport": [0.05, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.15],  # Alumni, industrialist high
    "talk": [
        0.2,
        0.05,
        0.1,
        0.05,
        0.15,
        0.1,
        0.15,
        0.15,
    ],  # Political, official, faculty high
    "presentation": [
        0.1,
        0.05,
        0.15,
        0.1,
        0.1,
        0.1,
        0.2,
        0.1,
    ],  # Faculty, professional high
    "gaming": [0.05, 0.25, 0.1, 0.2, 0.05, 0.1, 0.05, 0.1],  # YouTuber, influencer high
    "social service": [
        0.25,
        0.05,
        0.05,
        0.05,
        0.2,
        0.15,
        0.1,
        0.1,
    ],  # Political, official high
    "educational": [
        0.05,
        0.05,
        0.1,
        0.05,
        0.1,
        0.15,
        0.25,
        0.1,
    ],  # Faculty, alumni high
    "others": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Balanced
}

DEPARTMENTS = ["CSE", "ECE", "ME", "MM", "EE", "CE", "ECM", "PIE"]
DEPARTMENT_WEIGHTS = [0.25, 0.2, 0.1, 0.08, 0.08, 0.08, 0.08, 0.05]  # CSE most common

TIMINGS = ["Morning", "Afternoon", "Evening", "Late Night"]
TIMING_WEIGHTS = [0.3, 0.1, 0.55, 0.05]  # Evening most common

DAY_TYPES = ["Weekday", "Weekend"]
DAY_WEIGHTS = [0.4, 0.6]  # Weekends more common

WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Stormy"]
WEATHER_WEIGHTS = [0.55, 0.3, 0.1, 0.05]  # Sunny most common

# Score formula weights (carefully designed for realistic ranges: base 30-40 for hackathons, up to 600-700 for popular guests)
SCORE_WEIGHTS = {
    "event_type_score": 10.0,  # Base from event
    "guest_score": 25.0,  # Strong influence from guest (youtuber boosts high)
    "department_score": 5.0,
    "timing_score": 2.0,
    "day_score": 3.0,
    "promotion_level": 0.0002,  # Scaled down for gradual increase
    "event_duration": -0.5,
    "venue_capacity": 0.01,  # Less direct impact
    "registration_fee": -0.02,
    "social_media_buzz": 0.009,
    "concurrent_events_count": -3.0,
    "weather_score": 2.0,
}

# Base scores for categoricals (realistic: hackathon base ~30, youtuber boost high)
EVENT_TYPE_SCORES = {
    "hackathon": 130,
    "workshop": 350,
    "seminar": 40,
    "contest": 32,
    "fun": 100,
    "sport": 210,
    "talk": 60,
    "presentation": 36,
    "gaming": 100,
    "social service": 50,
    "educational": 25,
    "others": 30,
}

GUEST_SCORES = {
    "political": 300,
    "youtuber": 410,
    "professional": 100,
    "influencer": 100,
    "official": 30,
    "alumni": 60,
    "faculty": 35,
    "industrialist": 45,
}

DEPARTMENT_SCORES = {
    "CSE": 40,
    "ECE": 38,
    "ME": 30,
    "MM": 28,
    "EE": 35,
    "CE": 25,
    "ECM": 32,
    "PIE": 27,
}

TIMING_SCORES = {"Morning": 35, "Afternoon": 20, "Evening": 60, "Late Night": 50}

DAY_SCORES = {"Weekday": 10, "Weekend": 90}

WEATHER_SCORES = {"Sunny": 40, "Cloudy": 50, "Rainy": 10, "Stormy": 5}

# Score bias and noise (for realistic ranges)
SCORE_BIAS = 20.0
SCORE_NOISE_STD = 30.0

# Outlier probabilities
OUTLIER_PROB = 0.03

# Null probabilities
NULL_PROBS = {
    "promotion_level": (0.02, 0.05),
    "event_duration": (0.01, 0.03),
    "venue_capacity": (0.01, 0.03),
    "registration_fee": (0.02, 0.04),
    "social_media_buzz": (0.03, 0.06),
    "concurrent_events_count": (0.01, 0.02),
}

# ========================================
# DATA GENERATION
# ========================================

data = []

for i in range(TOTAL):
    # Categorical selections (correlated guest)
    event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS, k=1)[0]
    guest = random.choices(GUESTS, weights=GUEST_WEIGHTS_PER_EVENT[event_type], k=1)[0]
    organising_department = random.choices(
        DEPARTMENTS, weights=DEPARTMENT_WEIGHTS, k=1
    )[0]
    timing = random.choices(TIMINGS, weights=TIMING_WEIGHTS, k=1)[0]
    day_of_week = random.choices(DAY_TYPES, weights=DAY_WEIGHTS, k=1)[0]
    weather_condition = random.choices(
        WEATHER_CONDITIONS, weights=WEATHER_WEIGHTS, k=1
    )[0]

    # Numerical features (duration as integer)
    promotion_level = int(np.random.uniform(0, 1000))
    event_duration = max(
        0.5, min(12, int(np.random.lognormal(1.26, 0.4)))
    )  # 0.5-12 hrs
    venue_capacity = int(
        np.random.lognormal(np.log(200), 0.6)
    )  # typical 100–500, no negatives
    # Registration fee with guest popularity correlation
    base_fee_options = [
        50,
        100,
        120,
        150,
        180,
        200,
        210,
        250,
        300,
        350,
        360,
        400,
        420,
        450,
        500,
    ]
    registration_fee = int(np.random.choice(base_fee_options))
    # Premium fees for popular guests
    if guest in ["youtuber", "political"]:
        registration_fee += random.choice([20, 30, 40, 50])
    elif guest in ["influencer", "professional"]:
        registration_fee += random.choice([10, 15, 20])
    # Cap at maximum reasonable fee
    registration_fee = min(registration_fee, 500)
    social_media_buzz = int(
        np.random.lognormal(np.log(500), 1)
    )  # heavy-tailed, viral events exist
    concurrent_events_count = int(np.random.poisson(0.1))  # usually 0, very rarely 1

    # Scores
    event_type_score = EVENT_TYPE_SCORES[event_type]
    guest_score = GUEST_SCORES[guest]
    department_score = DEPARTMENT_SCORES[organising_department]
    timing_score = TIMING_SCORES[timing]
    day_score = DAY_SCORES[day_of_week]
    weather_score = WEATHER_SCORES[weather_condition]

    # Participant count formula
    base_score = (
        SCORE_WEIGHTS["event_type_score"] * (event_type_score / 100)
        + SCORE_WEIGHTS["guest_score"] * (guest_score / 100)
        + SCORE_WEIGHTS["department_score"] * (department_score / 100)
        + SCORE_WEIGHTS["timing_score"] * (timing_score / 100)
        + SCORE_WEIGHTS["day_score"] * (day_score / 100)
        + SCORE_WEIGHTS["promotion_level"] * promotion_level
        + SCORE_WEIGHTS["event_duration"] * event_duration
        + SCORE_WEIGHTS["venue_capacity"] * venue_capacity
        + SCORE_WEIGHTS["registration_fee"] * registration_fee
        - (registration_fee**2) / 5000  # Non-linear fee impact
        + SCORE_WEIGHTS["social_media_buzz"] * social_media_buzz
        + (promotion_level * np.log1p(social_media_buzz)) / 20  # Interaction
        + (event_duration * timing_score) / 50  # Duration x timing interaction
        + (venue_capacity * department_score) / 1000  # Capacity x dept interaction
        - (concurrent_events_count * registration_fee)
        / 100  # Concurrent x fee interaction
        + SCORE_WEIGHTS["concurrent_events_count"] * concurrent_events_count
        + SCORE_WEIGHTS["weather_score"] * (weather_score / 100)
        + SCORE_BIAS
    )

    participant_count = int(base_score)  # raw count without noise

    # Diverse outliers (before capacity bound)
    if random.random() < OUTLIER_PROB:
        # Viral events: high buzz + popular guest
        if social_media_buzz > 2000 and guest in ["youtuber", "influencer"]:
            participant_count = random.choice([1000, 1200, 1500])
        # Flops: bad weather + high fee + concurrent events
        elif (
            weather_condition in ["Rainy", "Stormy"]
            and registration_fee > 300
            and concurrent_events_count > 1
        ):
            participant_count = random.choice([10, 20, 30, 40])
        else:
            participant_count = random.choice([10, 700])

    # Ensure participant count is non-negative
    participant_count = max(participant_count, 0)

    # Bound attendance to venue capacity
    participant_count = min(participant_count, venue_capacity)
    participant_count = max(participant_count, 0)

    # Nulls
    for field, (low, high) in NULL_PROBS.items():
        if random.random() < random.uniform(low, high):
            locals()[field] = np.nan

    data.append(
        {
            "id": str(i + 1).zfill(3),
            "event_type": event_type,
            "guest": guest,
            "organising_department": organising_department,
            "timing": timing,
            "day_of_week": day_of_week,
            "promotion_level": promotion_level
            if not np.isnan(promotion_level)
            else np.nan,
            "event_duration": event_duration
            if not np.isnan(event_duration)
            else np.nan,
            "venue_capacity": venue_capacity
            if not np.isnan(venue_capacity)
            else np.nan,
            "registration_fee": registration_fee
            if not np.isnan(registration_fee)
            else np.nan,
            "social_media_buzz": social_media_buzz
            if not np.isnan(social_media_buzz)
            else np.nan,
            "concurrent_events_count": concurrent_events_count
            if not np.isnan(concurrent_events_count)
            else np.nan,
            "weather_condition": weather_condition,
            "participant_count": participant_count,
        }
    )

# ========================================
# SAVE DATASETS
# ========================================

df = pd.DataFrame(data)
train_df = df[:N_TRAIN]
eval_df = df[N_TRAIN:]

train_df.to_csv("train.csv", index=False)
eval_df.drop("participant_count", axis=1).to_csv("eval.csv", index=False)
eval_df[["participant_count"]].to_csv("eval_labels.csv", index=False)

print(
    f"Datasets generated: train.csv ({N_TRAIN}), eval.csv ({N_EVAL} features), eval_labels.csv ({N_EVAL} labels)"
)
