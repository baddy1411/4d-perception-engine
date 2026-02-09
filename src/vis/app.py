import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import argparse
import os

# Set environment variables for offline mode to prevent plugin lookup errors
os.environ["FIFTYONE_OFFLINE"] = "1"


def launch_dashboard(dataset_name="4d_perception_demo"):
    # Check if dataset exists, if so delete it for fresh start
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    
    # Load 50 samples from Open Images V7 validation split
    # We filter for "Car" to ensure relevance to autonomous driving
    print("Downloading 50 samples from Open Images V7 (Validation)...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        max_samples=50,
        label_types=["detections"],
        classes=["Car", "Bus", "Truck", "Man", "Woman"],
        dataset_name=dataset_name,
    )
    
    print("Generating metadata for perception demo...")
    # Add our custom perception fields to the real data
    for sample in dataset:
        # Add scalar fields
        sample["weather"] = np.random.choice(["clear", "rain", "fog", "snow"])
        sample["time_of_day"] = np.random.choice(["day", "night"])
        
        # Add derived metrics (simulating the 'Hard Example Miner' output)
        difficulty_score = np.random.beta(2, 5) # Skewed distribution
        sample["difficulty_score"] = difficulty_score
        
        # Tag hard examples
        if difficulty_score > 0.7 or sample["weather"] in ["rain", "snow"]:
            sample.tags.append("hard_example")
        
        sample.save()

    
    dataset.persistent = True
    dataset.save()
    
    # Launch App
    session = fo.launch_app(dataset, address="0.0.0.0", port=5152, auto=False)
    
    print("Session launched. Access at http://localhost:5152")
    session.wait()

if __name__ == "__main__":
    launch_dashboard()
