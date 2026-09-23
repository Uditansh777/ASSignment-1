import os
import re
import pandas as pd
import geopandas as gpd

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def setup_directories():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def parse_ratings(val):
    if val == "Unknown": return pd.Series([None, None])
    val = val.lower().replace(" ", "")
    matches = re.findall(r"(?:(\d+)x)?(\d+)kw", val)
    if not matches:
        nums = re.findall(r"\d+", val)
        if nums:
            max_kw = float(max(map(int, nums)))
            return pd.Series([max_kw, max_kw])
        return pd.Series([None, None])
    max_kw = 0
    total_kw = 0
    for count_str, kw_str in matches:
        count = int(count_str) if count_str else 1
        kw = int(kw_str)
        max_kw = max(max_kw, kw)
        total_kw += count * kw
    return pd.Series([max_kw, total_kw])

def load_and_clean_ev_data():
    print("Loading EV data...")
    ev_file = os.path.join(RAW_DIR, "ev_charging_locations.csv")
    df = pd.read_csv(ev_file)
    
    print(f"Original EV records: {len(df)}")
    
    lat_col = "Latitude" if "Latitude" in df.columns else "latitude" if "latitude" in df.columns else None
    lon_col = "Longitude" if "Longitude" in df.columns else "longitude" if "longitude" in df.columns else None
    
    if not lat_col or not lon_col:
        lat_col = df.columns[df.columns.str.lower().str.contains("lat")][0]
        lon_col = df.columns[df.columns.str.lower().str.contains("lon")][0]

    df = df.dropna(subset=[lat_col, lon_col])
    
    df = df.drop_duplicates()
    
    if "OBJECTID" in df.columns:
        df = df.drop(columns=["OBJECTID"])
    
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col].str.lower() == "nan", col] = "Unknown"
        df.loc[df[col].str.lower() == "<na>", col] = "Unknown"
        df.loc[df[col] == "", col] = "Unknown"
        
    if "Operator" in df.columns:
        df["Operator"] = df["Operator"].str.replace("NRMA Parks and Resorts", "NRMA")
        operator_mapping = {
            "ChargePoint": "Chargefox", 
            "EVE Australia": "EVX", 
            "Non-networked": "Unknown"
        }
        df["Operator"] = df["Operator"].replace(operator_mapping)
        
    if "Charger_Type" in df.columns:
        df["Status"] = "Operational"
        df.loc[df["Charger_Type"].str.lower() == "upcoming", "Status"] = "Upcoming"
        df.loc[df["Charger_Type"].str.lower() == "upcoming", "Charger_Type"] = "Unknown"
        
    if "Charger_rating" in df.columns:
        df.loc[df["Charger_rating"] == "AC", "Charger_rating"] = "Unknown"
        df[["Max_kW_Rating", "Total_kW_Capacity"]] = df["Charger_rating"].apply(parse_ratings)
        
    print(f"Cleaned EV records: {len(df)}")
    
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )
    return gdf

def load_sa4_boundaries():
    print("Loading SA4 boundaries...")
    sa4_dir = os.path.join(RAW_DIR, "sa4_boundaries")
    
    spatial_file = None
    for root, dirs, files in os.walk(sa4_dir):
        for file in files:
            if file.endswith(".shp") or file.endswith(".gpkg"):
                spatial_file = os.path.join(root, file)
                break
        if spatial_file:
            break
            
    if not spatial_file:
        raise FileNotFoundError("Could not find a .shp or .gpkg file in the extracted SA4 boundaries.")
        
    sa4_gdf = gpd.read_file(spatial_file)
    return sa4_gdf

def perform_spatial_join(ev_gdf, sa4_gdf):
    print("Performing spatial join...")
    
    if ev_gdf.crs != sa4_gdf.crs:
        print(f"Reprojecting EV data to match SA4 CRS ({sa4_gdf.crs})...")
        ev_gdf = ev_gdf.to_crs(sa4_gdf.crs)
        
    joined_gdf = gpd.sjoin(ev_gdf, sa4_gdf, how="left", predicate="within")
    
    if "SA4_NAME26" in joined_gdf.columns:
        orphan_mask = joined_gdf["SA4_NAME26"].isna()
        if orphan_mask.any():
            geo_cols = [c for c in joined_gdf.columns if "26" in c or "index" in c or c.startswith("CHG_")]
            for col in geo_cols:
                if joined_gdf[col].dtype == 'object' or joined_gdf[col].dtype == 'string':
                    joined_gdf.loc[orphan_mask, col] = "Out of Bounds"
    
    print(f"Joined records: {len(joined_gdf)}")
    return joined_gdf

if __name__ == "__main__":
    setup_directories()
    
    try:
        ev_gdf = load_and_clean_ev_data()
        sa4_gdf = load_sa4_boundaries()
        
        final_gdf = perform_spatial_join(ev_gdf, sa4_gdf)
        
        output_file = os.path.join(PROCESSED_DIR, "ev_chargers_integrated.csv")
        final_df = pd.DataFrame(final_gdf.drop(columns="geometry"))
        final_df.to_csv(output_file, index=False)
        print(f"Successfully saved integrated data to {output_file}")
        
    except Exception as e:
        print(f"Error during integration: {e}")
