import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import argparse

def launch_dashboard(dataset_name="4d_perception_demo"):
    # Check if dataset exists, if so delete it for fresh start
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    
    dataset = fo.Dataset(dataset_name)
    
    print("Generating mock perception data for visualization...")
    
    # Create mock samples
    samples = []
    for i in range(50):
        # Simulate an image path (in reality, these would be local paths or S3 URLs)
        # Using a placeholder image from web for demo purposes if we were scraping, 
        # but here we just create a sample object.
        sample = fo.Sample(filepath=f"/data/camera_front/{i:06d}.jpg")
        
        # Add scalar fields
        sample["weather"] = np.random.choice(["clear", "rain", "fog", "snow"])
        sample["time_of_day"] = np.random.choice(["day", "night"])
        
        # Add derived metrics (simulating the 'Hard Example Miner' output)
        difficulty_score = np.random.beta(2, 5) # Skewed distribution
        sample["difficulty_score"] = difficulty_score
        
        # Tag hard examples
        if difficulty_score > 0.7 or sample["weather"] in ["rain", "snow"]:
            sample.tags.append("hard_example")
            
        # Add detections (simulated)
        detections = []
        for _ in range(np.random.randint(0, 5)):
            detections.append(
                fo.Detection(
                    label=np.random.choice(["car", "pedestrian", "cyclist"]),
                    bounding_box=[
                        np.random.rand()*0.8,
                        np.random.rand()*0.8,
                        np.random.rand()*0.2,
                        np.random.rand()*0.2
                    ],
                    confidence=np.random.uniform(0.5, 0.99)
                )
            )
        sample["predictions"] = fo.Detections(detections=detections)
        
        samples.append(sample)

    dataset.add_samples(samples)
    
    print(f"Dataset '{dataset_name}' created with {len(samples)} samples.")
    
    # Launch App
    session = fo.launch_app(dataset, port=5151)
    
    print("Session launched. Access at localhost:5151")
    session.wait()

if __name__ == "__main__":
    launch_dashboard()
