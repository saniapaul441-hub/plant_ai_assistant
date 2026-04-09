import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = "dataset/PlantVillage"
MODEL_DIR = "model"
JSON_PATH = os.path.join(MODEL_DIR, "class_names.json")

def verify_and_save_labels():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' does not exist.")
        return

    print(f"Reading dataset from {DATASET_PATH}...")
    
    # We match the ImageDataGenerator logic from train_model.py
    datagen = ImageDataGenerator()
    train_data = datagen.flow_from_directory(
        DATASET_PATH,
        # Target size doesn't matter for just reading class indices
        target_size=(224, 224), 
        batch_size=1,
        class_mode="categorical",
        shuffle=False
    )
    
    # Keras flow_from_directory outputs class_indices as a dict: {"class_name": index}
    class_indices = train_data.class_indices
    
    # We need to map index -> class_name for detection_agent.py
    index_to_class = {int(v): k for k, v in class_indices.items()}
    
    # Verify the alphabetical order
    folders = sorted([x for x in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, x))])
    
    print("\n--- Label Mapping Verification ---")
    verification_passed = True
    for v in sorted(index_to_class.keys()):
        keras_class = index_to_class[v]
        folder_class = folders[v] if v < len(folders) else "Unknown"
        
        match = "[OK]" if keras_class == folder_class else "[MISMATCH!]"
        if keras_class != folder_class:
            verification_passed = False
            
        print(f"Index {v:2d}: Keras='{keras_class}' | Folder='{folder_class}' {match}")

    if verification_passed:
        print("\nAll mappings perfectly match the alphabetical folder order!")
    else:
        print("\nMismatch detected! The Keras indices differ from raw os.listdir order.")
        
    # Save the reliable Keras mapping to JSON
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(index_to_class, f, indent=4)
        
    print(f"Saved correct index-to-class mapping to: {JSON_PATH}")
    print("Your inference script will now load these exact labels and give correct outputs.")

if __name__ == "__main__":
    verify_and_save_labels()
