"""Shared configuration: column names, paths and experiment defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CITIES = ("nyc", "paris", "berlin")

SEED = 42
PRICE_CAP = 500            # data thresholding: drop listings above $500/night
SPLIT = (0.7, 0.2, 0.1)    # train : validation : test

# --- continuous features (only those present in a given CSV are used) -------
MONEY_COLS = ["security_deposit", "cleaning_fee", "extra_people"]
ZERO_FILL_COLS = ["security_deposit", "cleaning_fee"]   # missing fee == no fee
CONTINUOUS_COLS = [
    "accommodates", "bathrooms", "bedrooms", "beds", "guests_included",
    "security_deposit", "cleaning_fee", "extra_people",
    "minimum_nights", "maximum_nights",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "reviews_per_month",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
    "latitude", "longitude",
    "host_listings_count", "host_total_listings_count",
    "calculated_host_listings_count",
]
CORR_THRESHOLD = 0.9   # drop one of each pair of continuous features above this

# --- categorical features (one-hot) -----------------------------------------
CATEGORICAL_COLS = [
    "property_type", "room_type", "bed_type", "cancellation_policy",
    "neighbourhood_cleansed", "neighbourhood_group_cleansed", "city",
    "host_is_superhost", "host_identity_verified", "instant_bookable",
    "is_location_exact", "require_guest_profile_picture",
    "require_guest_phone_verification", "host_has_profile_pic",
    "host_response_time", "requires_license", "is_business_travel_ready",
]
# list-valued columns encoded via dictionary building (multi-hot)
LIST_COLS = {"amenities": 40, "host_verifications": 20}   # column -> top-N tokens

# --- text features (TF-IDF unigram+bigram -> truncated SVD) -----------------
TEXT_COLS = [
    "name", "summary", "space", "description", "neighborhood_overview",
    "notes", "transit", "access", "interaction", "house_rules", "host_about",
    "host_neighbourhood",
]
TFIDF_MAX_FEATURES = 20000
SVD_COMPONENTS = 50

# --- date features -----------------------------------------------------------
DATE_COLS = ["host_since", "first_review", "last_review"]
