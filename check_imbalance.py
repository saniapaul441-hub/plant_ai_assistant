import os

dataset_path = "dataset/PlantVillage"
if os.path.exists(dataset_path):
    folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
    
    print("Class Distribution:")
    for f in folders:
        path = os.path.join(dataset_path, f)
        count = len(os.listdir(path))
        print(f"{f}: {count} images")
