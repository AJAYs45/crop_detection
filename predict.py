import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os
import urllib.parse # 🚀 NEW

# --- Settings ---
MODEL_PATH = 'crop_model.h5'
LABELS_PATH = 'class_indices.json'

print("Loading the trained model... Please wait.")

if not os.path.exists(MODEL_PATH):
    print("Error: Model not found. Please ensure 'train.py' has finished running.")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    class_names = json.load(f)
    class_names = {int(k): v for k, v in class_names.items()}

def predict_disease(img_path):
    if not os.path.exists(img_path):
        print("Error: The specified image file was not found.")
        return

    # Image Preprocessing (matching the training format)
    # Note: Keep target_size=(224,224) if your model was trained on 224x224
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    # 🚀 NEW FEATURE: Invalid Image
    if confidence < 70.0:
        print("\n[!] Error: Invalid Image. Please upload a clear crop leaf photo.")
        return
    
    disease_name = class_names[predicted_index]
    
    # 🚀 NEW FEATURE: Wiki URL
    clean_name = disease_name.replace("___", " ").replace("_", " ")
    wiki_url = f"https://en.wikipedia.org/w/index.php?search={urllib.parse.quote(clean_name)}"

    print("\n" + "="*50)
    print(f"Image Path  : {img_path}")
    print(f"Prediction  : {disease_name}")
    print(f"Confidence  : {confidence:.2f}%")
    print(f"Wiki Link   : {wiki_url}")
    print("="*50 + "\n")

if __name__ == "__main__":
    print("Ready for prediction!")
    path = input("Enter the path or name of the image (e.g., test_leaf.jpg): ")
    predict_disease(path.strip().replace('"', '').replace("'", ""))