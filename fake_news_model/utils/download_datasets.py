import os
import shutil
import requests
import zipfile
import json
import pandas as pd
from tqdm import tqdm

def download_file(url, filename):
    """
    Download a file from a URL with progress bar
    """
    print(f"Downloading {filename} from {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

def download_fever_dataset():
    """
    Download the FEVER dataset
    """
    base_dir = "data/raw/fever"
    os.makedirs(base_dir, exist_ok=True)
    
    # FEVER dataset links
    fever_train_url = "https://fever.ai/download/fever/train.jsonl"
    fever_dev_url = "https://fever.ai/download/fever/dev.jsonl"
    
    # Download files
    download_file(fever_train_url, os.path.join(base_dir, "train.jsonl"))
    download_file(fever_dev_url, os.path.join(base_dir, "dev.jsonl"))
    
    print("FEVER dataset downloaded successfully.")

def download_liar_dataset():
    """
    Download the LIAR dataset
    """
    base_dir = "data/raw/liar"
    os.makedirs(base_dir, exist_ok=True)
    
    # LIAR dataset link
    liar_url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
    zip_path = os.path.join(base_dir, "liar_dataset.zip")
    
    # Download and extract
    download_file(liar_url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)
    
    # Rename files for consistency
    if os.path.exists(os.path.join(base_dir, "train.tsv")):
        print("LIAR dataset files already in place.")
    else:
        # Move files from the extracted directory to our base directory
        extracted_dir = os.path.join(base_dir, "liar_dataset")
        for file in os.listdir(extracted_dir):
            shutil.move(os.path.join(extracted_dir, file), os.path.join(base_dir, file))
        
        # Clean up
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    print("LIAR dataset downloaded and extracted successfully.")

def create_mock_politifact_dataset():
    """
    Create a small mock PolitiFact dataset for testing
    """
    base_dir = "data/raw/politifact"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a sample dataset
    data = {
        'statement': [
            "The COVID-19 vaccine contains microchips that track people.",
            "Global temperatures have risen by 1.1°C since pre-industrial times.",
            "The Earth is flat, not round.",
            "Drinking water with lemon every morning cures cancer.",
            "The Apollo moon landings were faked.",
            "5G towers cause COVID-19.",
            "Vaccines cause autism."
        ],
        'rating': [
            "false",
            "true",
            "pants-fire",
            "false",
            "pants-fire",
            "pants-fire",
            "false"
        ],
        'context': [
            "Claims about COVID-19 vaccines containing tracking microchips have spread on social media.",
            "NASA and NOAA data has shown approximately 1.1°C warming since pre-industrial times.",
            "Flat Earth claims persist despite centuries of scientific evidence.",
            "Lemon water health claims have circulated widely on social media.",
            "Conspiracy theories about the moon landing being filmed on Earth have persisted for decades.",
            "Claims linking 5G technology to the spread of COVID-19 emerged in early 2020.",
            "The original study claiming a link between vaccines and autism was retracted and proven fraudulent."
        ],
        'justification': [
            "There is no evidence that COVID-19 vaccines contain microchips. Vaccines contain ingredients to trigger an immune response but no tracking devices.",
            "Temperature records from NASA, NOAA, and other agencies confirm this level of warming based on global surface temperature measurements.",
            "The Earth's curvature has been measured in numerous ways, from ancient Greek calculations to modern satellite imagery and space travel.",
            "While lemons contain vitamin C and antioxidants, there is no scientific evidence they can cure cancer. Cancer requires proper medical treatment.",
            "Multiple independent countries have verified the moon landings, plus we have moon rocks, reflectors placed by astronauts, and landing site photos.",
            "COVID-19 is caused by a virus that is unrelated to electromagnetic radiation. 5G technology has been tested and meets international safety standards.",
            "Multiple large-scale studies involving millions of children have found no link between vaccines and autism."
        ]
    }
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(base_dir, "politifact_data.csv"), index=False)
    
    print("Mock PolitiFact dataset created successfully.")

def main():
    """
    Main function to download all datasets
    """
    print("Starting dataset downloads...")
    
    try:
        download_fever_dataset()
    except Exception as e:
        print(f"Error downloading FEVER dataset: {e}")
        print("Creating placeholder file instead.")
        os.makedirs("data/raw/fever", exist_ok=True)
        with open("data/raw/fever/train.jsonl", 'w') as f:
            f.write(json.dumps({"id": 0, "label": "SUPPORTS", "claim": "This is a sample claim", "evidence": [["0", "Sample", "Sample", "This is sample evidence."]]}) + "\n")
    
    try:
        download_liar_dataset()
    except Exception as e:
        print(f"Error downloading LIAR dataset: {e}")
        print("Creating placeholder file instead.")
        os.makedirs("data/raw/liar", exist_ok=True)
        with open("data/raw/liar/train.tsv", 'w') as f:
            f.write("0\ttrue\tThis is a sample true claim.\tSample context\tSample justification\n")
    
    try:
        create_mock_politifact_dataset()
    except Exception as e:
        print(f"Error creating mock PolitiFact dataset: {e}")
    
    print("\nAll datasets processed. Ready for model training.")

if __name__ == "__main__":
    main() 