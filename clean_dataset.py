import os
from PIL import Image

dataset_path = "dataset/PlantVillage"
removed = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)
                removed += 1

print(f"Removed {removed} corrupted images.")
