import os
import requests
import zipfile
from bs4 import BeautifulSoup

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")

def setup_directories():
    os.makedirs(RAW_DIR, exist_ok=True)

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

def get_ev_data():
    # Attempting to get data from data.gov.au CKAN API
    print("Fetching EV charging locations from data.gov.au...")
    api_url = "https://data.gov.au/data/api/3/action/package_show?id=nsw-2-ev-charging-locations"
    res = requests.get(api_url)
    res.raise_for_status()
    data = res.json()
    
    # Find the CSV resource
    csv_url = None
    for resource in data['result']['resources']:
        if resource['format'].lower() == 'csv':
            csv_url = resource['url']
            break
            
    if not csv_url:
        raise ValueError("Could not find CSV download URL for EV charging locations.")
        
    dest = os.path.join(RAW_DIR, "ev_charging_locations.csv")
    download_file(csv_url, dest)

def get_abs_data():
    print("Fetching ASGS SA4 boundaries from ABS...")
    # The assignment gives this URL:
    abs_page_url = "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-4-july-2026-june-2031/access-and-downloads/digital-boundary-files"
    
    # We will scrape the page to find the SA4 shapefile download link
    res = requests.get(abs_page_url)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'html.parser')
    
    # Look for a link that mentions SA4 and shapefile/gpkg
    download_link = None
    for a in soup.find_all('a', href=True):
        text = a.text.lower()
        href = a.get('href', '')
        if 'sa4' in text or 'sa4' in href.lower():
            if '.zip' in href.lower() or 'shapefile' in text or 'gpkg' in text or 'esri' in text:
                download_link = href
                if not download_link.startswith('http'):
                    download_link = "https://www.abs.gov.au" + download_link
                break
                
    # Fallback if scraping fails
    if not download_link:
        print("Warning: Could not automatically find the SA4 shapefile link on the ABS page.")
        print("Please provide the direct .zip URL for the SA4 shapefile:")
        download_link = input("URL: ").strip()
        
    zip_dest = os.path.join(RAW_DIR, "sa4_boundaries.zip")
    download_file(download_link, zip_dest)
    
    # Extract the zip file
    print(f"Extracting {zip_dest}...")
    with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(RAW_DIR, "sa4_boundaries"))
    print("Extraction complete.")

if __name__ == "__main__":
    setup_directories()
    try:
        get_ev_data()
    except Exception as e:
        print(f"Error fetching EV data: {e}")
        
    try:
        get_abs_data()
    except Exception as e:
        print(f"Error fetching ABS data: {e}")
