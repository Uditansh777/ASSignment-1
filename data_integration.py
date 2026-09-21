import os
import pandas as pd
import geopandas as gpd

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def setup_directories():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_and_clean_ev_data():
    print("Loading EV data...")
    ev_file = os.path.join(RAW_DIR, "ev_charging_locations.csv")
    df = pd.read_csv(ev_file)
    
    print(f"Original EV records: {len(df)}")
    
    # 1. Address missing coordinates (crucial for spatial join)
    lat_col = 'Latitude' if 'Latitude' in df.columns else 'latitude' if 'latitude' in df.columns else None
    lon_col = 'Longitude' if 'Longitude' in df.columns else 'longitude' if 'longitude' in df.columns else None
    
    if not lat_col or not lon_col:
        # Fallback if names are completely different
        lat_col = df.columns[df.columns.str.lower().str.contains('lat')][0]
        lon_col = df.columns[df.columns.str.lower().str.contains('lon')][0]

    df = df.dropna(subset=[lat_col, lon_col])
    
    # 2. Drop exact duplicates
    df = df.drop_duplicates()
    
    # 3. Clean string columns (Generic)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].str.lower() == 'nan', col] = "Unknown"
        
    # 4. Fix specific inconsistent Operator naming
    if 'Operator' in df.columns:
        # Standardize known variations or typos in operator names
        operator_mapping = {
            'ChargePoint': 'Chargefox', # Example consolidation
            'EVE Australia': 'EVX', # Example typo fix
            'Non-networked': 'Unknown'
        }
        df['Operator'] = df['Operator'].replace(operator_mapping)
        
    # 5. Fix inconsistent Charger attributes
    if 'Charger_rating' in df.columns:
        # Some rows have 'AC' instead of a kilowatt rating (e.g. '22 kW')
        # Let's standardize this to 'Unknown' where it's not a proper kW rating
        df.loc[df['Charger_rating'] == 'AC', 'Charger_rating'] = 'Unknown'
        
    print(f"Cleaned EV records: {len(df)}")
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326" # GPS coordinates are WGS84
    )
    return gdf

def load_sa4_boundaries():
    print("Loading SA4 boundaries...")
    sa4_dir = os.path.join(RAW_DIR, "sa4_boundaries")
    
    # Find the shapefile (.shp) or geopackage (.gpkg)
    spatial_file = None
    for root, dirs, files in os.walk(sa4_dir):
        for file in files:
            if file.endswith('.shp') or file.endswith('.gpkg'):
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
    
    print(f"EV CRS: {ev_gdf.crs}")
    print(f"SA4 CRS: {sa4_gdf.crs}")
    
    # Align Coordinate Reference Systems
    if ev_gdf.crs != sa4_gdf.crs:
        print(f"Reprojecting EV data to match SA4 CRS ({sa4_gdf.crs})...")
        ev_gdf = ev_gdf.to_crs(sa4_gdf.crs)
        
    # Spatially join the points to the polygons
    joined_gdf = gpd.sjoin(ev_gdf, sa4_gdf, how="left", predicate="within")
    
    print(f"Joined records: {len(joined_gdf)}")
    return joined_gdf

if __name__ == "__main__":
    setup_directories()
    
    try:
        ev_gdf = load_and_clean_ev_data()
        sa4_gdf = load_sa4_boundaries()
        
        final_gdf = perform_spatial_join(ev_gdf, sa4_gdf)
        
        output_file = os.path.join(PROCESSED_DIR, "ev_chargers_integrated.csv")
        # Drop the geometry column for CSV storage as it's not needed for simple CSV
        final_df = pd.DataFrame(final_gdf.drop(columns='geometry'))
        final_df.to_csv(output_file, index=False)
        print(f"Successfully saved integrated data to {output_file}")
        
    except Exception as e:
        print(f"Error during integration: {e}")
