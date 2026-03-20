import requests
import zipfile
import io
import os

EXTRACTION_PATH = r"g:\ML\Cricket\Data\India_Specific"

DATA_URLS = {
    "IPL": "https://cricsheet.org/downloads/ipl_csv2.zip",
    "ODI": "https://cricsheet.org/downloads/odis_csv2.zip",
    "TEST": "https://cricsheet.org/downloads/tests_csv2.zip"
}

def download_and_extract():
    if not os.path.exists(EXTRACTION_PATH):
        os.makedirs(EXTRACTION_PATH)
        
    for name, url in DATA_URLS.items():
        print(f"Downloading {name} data from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            extracted_dir = os.path.join(EXTRACTION_PATH, name)
            if not os.path.exists(extracted_dir):
                os.makedirs(extracted_dir)
                
            print(f"Download complete. Extracting {name} files...")
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(extracted_dir)
                
            print(f"Extracted flawlessly to {extracted_dir}")
        except Exception as e:
            print(f"Failed to fetch {name}: {e}")

if __name__ == "__main__":
    download_and_extract()
