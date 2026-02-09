import fiftyone as fo

print("Existing datasets:", fo.list_datasets())

if "4d_perception_demo" in fo.list_datasets():
    dataset = fo.load_dataset("4d_perception_demo")
    print(f"Sample count: {len(dataset)}")
    print(f"Persistent: {dataset.persistent}")
else:
    print("Dataset not found!")
