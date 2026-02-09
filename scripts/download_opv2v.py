"""
Download OPV2V-H dataset from HuggingFace
Downloads a subset (~4-5GB) of real LiDAR data for the perception demo.
"""

from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path
import os
import zipfile
import shutil

REPO_ID = "yifanlu/OPV2V-H"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "opv2v"

def list_available_files():
    """List all files in the HuggingFace repo."""
    print(f"Listing files in {REPO_ID}...")
    files = list_repo_files(REPO_ID, repo_type="dataset")
    for f in files:
        print(f"  {f}")
    return files

def download_depth_data():
    """Download the depth camera data (~2GB)."""
    print("\n📥 Downloading OPV2V-H depth data...")
    try:
        filepath = hf_hub_download(
            repo_id=REPO_ID,
            filename="OPV2V-H-depth.zip",
            repo_type="dataset",
            local_dir=OUTPUT_DIR,
        )
        print(f"  Downloaded to: {filepath}")
        return filepath
    except Exception as e:
        print(f"  Error: {e}")
        return None

def download_lidar_parts(num_parts: int = 2):
    """Download first N parts of LiDAR data (~3GB for 2 parts)."""
    print(f"\n📥 Downloading OPV2V-H LiDAR data (first {num_parts} parts)...")
    downloaded = []
    
    for i in range(num_parts):
        part_name = f"OPV2V-H-LiDAR-part{i:02d}"
        print(f"  Downloading {part_name}...")
        try:
            filepath = hf_hub_download(
                repo_id=REPO_ID,
                filename=part_name,
                repo_type="dataset",
                local_dir=OUTPUT_DIR,
            )
            downloaded.append(filepath)
            print(f"    ✓ Downloaded: {filepath}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    return downloaded

def extract_zip(zip_path: Path, extract_to: Path):
    """Extract a zip file."""
    print(f"\n📦 Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        print(f"  ✓ Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # List available files
    files = list_available_files()
    
    # Download depth data (usually smaller, good for demo)
    depth_path = download_depth_data()
    
    # Download first 2 LiDAR parts (~3GB)
    lidar_paths = download_lidar_parts(num_parts=2)
    
    # Extract depth data if downloaded
    if depth_path and Path(depth_path).exists():
        extract_zip(Path(depth_path), OUTPUT_DIR)
    
    print("\n✅ Download complete!")
    print(f"Data saved to: {OUTPUT_DIR}")
    print("\nNote: LiDAR parts need to be combined before extraction:")
    print("  cat OPV2V-H-LiDAR-part* > OPV2V-H-LiDAR.zip")
    print("  unzip OPV2V-H-LiDAR.zip")

if __name__ == "__main__":
    main()
