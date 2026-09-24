
import os
import time
import json
import math
from pathlib import Path
from rapidfuzz import fuzz
import requests
import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = RAW_DIR / "ocm_cache"

INPUT_CSV = PROCESSED_DIR / "ev_chargers_integrated.csv"          # from data_integration.py
OCM_CACHE_FILE = CACHE_DIR / "ocm_raw_responses.json"              # local copy of raw API hits
OUTPUT_CSV = PROCESSED_DIR / "ev_chargers_dc_augmented.csv"

OCM_BASE_URL = "https://api.openchargemap.io/v3/poi/"
# Get a free key at https://openchargemap.org/site/loginprovider 
OCM_API_KEY = os.environ.get("OCM_API_KEY", "")


SEARCH_RADIUS_KM = 1.0          # how far to search around each charger for OCM candidates
DISTANCE_MATCH_THRESHOLD_KM = 1.0   # max distance to accept a coordinate match
NAME_SIMILARITY_THRESHOLD = 0.6     # min fuzzy-match score to accept a name match (0-1)


def setup_directories():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Step 1: Load the DC subset from the already-cleaned/integrated data
# --------------------------------------------------------------------------

def load_dc_chargers() -> pd.DataFrame:
    """Load ev_chargers_integrated.csv and return only Charger_Type == 'DC' rows."""
    df = pd.read_csv(INPUT_CSV)
    dc_df = df[df["Charger_Type"].str.upper() == "DC"].copy()
    print(f"Loaded {len(df)} total chargers, {len(dc_df)} are DC (fast) chargers.")
    return dc_df


# --------------------------------------------------------------------------
# Step 2: Query Open Charge Map around a coordinate, with local caching
# --------------------------------------------------------------------------

def _load_cache() -> dict:
    if OCM_CACHE_FILE.exists():
        with open(OCM_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    with open(OCM_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def fetch_ocm_nearby(lat: float, lon: float, cache: dict) -> list:
    """
    Query OCM for POIs within SEARCH_RADIUS_KM of (lat, lon).
    Results are cached locally by rounded-coordinate key so re-running the
    script doesn't re-hit the API (important given rate limits).

    Returns the raw list of OCM POI dicts (see OCM API docs for schema:
    https://openchargemap.org/site/develop/api#/operations/get-poi).
    """
    key = f"{round(lat, 5)},{round(lon, 5)}"
    if key in cache:
        return cache[key]

    params = {
        "output": "json",
        "latitude": lat,
        "longitude": lon,
        "distance": SEARCH_RADIUS_KM,
        "distanceunit": "KM",
        "maxresults": 5,
        "compact": "false",
        "verbose": "true",
    }
    if OCM_API_KEY:
        params["key"] = OCM_API_KEY

    try:
        resp = requests.get(OCM_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json()
    except requests.RequestException as e:
        print(f"  OCM request failed for {key}: {e}")
        results = []

    cache[key] = results
    time.sleep(0.5)  # be polite to the free API tier — tune as needed
    return results


# --------------------------------------------------------------------------
# Step 3: Distance helper 
# --------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points, in kilometres."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------
# Step 4: Matching logic 
# --------------------------------------------------------------------------

def match_station_to_ocm(row: pd.Series, ocm_candidates: list) -> dict | None:
    """
    Match one DC charger to the nearest Open Charge Map (OCM) POI
    using coordinate distance as the primary matching criterion.
    """

    if not ocm_candidates:
        return None

    station_lat = row["Latitude"]
    station_lon = row["Longitude"]

    candidates = []

    for poi in ocm_candidates:
        address_info = poi.get("AddressInfo") or {}

        ocm_lat = address_info.get("Latitude")
        ocm_lon = address_info.get("Longitude")

        if ocm_lat is None or ocm_lon is None:
            continue

        distance = haversine_km(
            float(station_lat),
            float(station_lon),
            float(ocm_lat),
            float(ocm_lon)
        )

        candidates.append({
            "poi": poi,
            "distance": distance,
        })

    if not candidates:
        return None

    # Select the geographically closest OCM POI
    best = min(candidates, key=lambda x: x["distance"])

    distance = best["distance"]
    poi = best["poi"]

    # Accept if the OCM POI is within 300 metres
    if distance > DISTANCE_MATCH_THRESHOLD_KM:
        return None

    # --------------------------------------------------------------
    # Extract OCM attributes
    # --------------------------------------------------------------

    connections = poi.get("Connections") or []

    plug_types = []

    for connection in connections:
        connection_type = connection.get("ConnectionType") or {}
        title = connection_type.get("Title")

        if title:
            plug_types.append(title)

    # Remove duplicates
    plug_types = list(dict.fromkeys(plug_types))

    # Number of charging points / bays
    num_bays = poi.get("NumberOfPoints")

    if num_bays is None:
        num_bays = len(connections) if connections else None

    # Operator
    operator_info = poi.get("OperatorInfo") or {}
    operator_ocm = operator_info.get("Title")

    # Pricing
    usage_cost = poi.get("UsageCost")

    return {
        "ocm_poi_id": poi.get("ID"),
        "plug_types": ", ".join(plug_types) if plug_types else None,
        "num_bays": num_bays,
        "operator_ocm": operator_ocm,
        "pricing_ocm": usage_cost,
        "match_distance_km": round(distance, 4),
        "name_similarity": None,
        "address_similarity": None,
        "match_score": None,
        "match_confidence": "coordinate_distance",
    }


# --------------------------------------------------------------------------
# Step 5: Orchestration — run the augmentation over all DC chargers
# --------------------------------------------------------------------------

def augment_dc_chargers(dc_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every DC charger:
      1. fetch nearby OCM candidates (cached)
      2. attempt to match
      3. attach additional OCM attributes
      4. calculate and report the match rate
    """

    cache = _load_cache()
    setup_directories()

    results = []

    for idx, row in dc_df.iterrows():

        print(
            f"Processing {idx + 1}/{len(dc_df)}..."
        )

        # --------------------------------------------------------------
        # 1. Query OCM
        # --------------------------------------------------------------
        candidates = fetch_ocm_nearby(
            row["Latitude"],
            row["Longitude"],
            cache
        )

        # --------------------------------------------------------------
        # 2. Match station
        # --------------------------------------------------------------
        match = match_station_to_ocm(
            row,
            candidates
        )

        # --------------------------------------------------------------
        # 3. Store result
        # --------------------------------------------------------------
        if match is None:

            results.append({
                "ocm_poi_id": None,
                "plug_types": None,
                "num_bays": None,
                "operator_ocm": None,
                "pricing_ocm": None,
                "match_distance_km": None,
                "name_similarity": None,
                "address_similarity": None,
                "match_score": None,
                "match_confidence": "no_match",
            })

        else:

            results.append(match)

            print(
                f"  Matched OCM POI {match['ocm_poi_id']} "
                f"({match['match_confidence']}, "
                f"{match['match_distance_km']:.3f} km)"
            )

    # --------------------------------------------------------------
    # 4. Save cache
    # --------------------------------------------------------------
    _save_cache(cache)

    # --------------------------------------------------------------
    # 5. Merge results back with DC dataframe
    # --------------------------------------------------------------
    result_df = pd.DataFrame(
        results,
        index=dc_df.index
    )

    augmented_df = pd.concat(
        [
            dc_df.reset_index(drop=True),
            result_df.reset_index(drop=True)
        ],
        axis=1
    )

    # --------------------------------------------------------------
    # 6. Calculate match rate
    # --------------------------------------------------------------
    matched = augmented_df["ocm_poi_id"].notna().sum()
    total = len(augmented_df)

    if total > 0:
        match_rate = matched / total
    else:
        match_rate = 0

    print("\n" + "=" * 60)
    print("OCM AUGMENTATION SUMMARY")
    print("=" * 60)

    print(
        f"Matched {matched}/{total} DC chargers "
        f"({match_rate:.1%})"
    )

    print(
        f"Target >= 50%: "
        f"{'PASS' if match_rate >= 0.50 else 'NOT REACHED'}"
    )

    # --------------------------------------------------------------
    # 7. Match confidence summary
    # --------------------------------------------------------------
    print("\nMatch confidence:")
    print(
        augmented_df["match_confidence"]
        .value_counts(dropna=False)
    )

    return augmented_df


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    setup_directories()
    dc_chargers = load_dc_chargers()
    augmented = augment_dc_chargers(dc_chargers)
    augmented.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved augmented DC charger data to {OUTPUT_CSV}")