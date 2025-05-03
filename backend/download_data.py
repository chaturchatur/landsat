import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def main():
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Download EuroSAT dataset
    url = 'http://madm.dfki.de/files/sentinel/EuroSAT.zip'
    zip_path = os.path.join(data_dir, 'EuroSAT.zip')
    
    print("Downloading EuroSAT dataset...")
    download_file(url, zip_path)
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Remove the zip file
    os.remove(zip_path)
    print("Download and extraction complete!")

if __name__ == '__main__':
    main() 