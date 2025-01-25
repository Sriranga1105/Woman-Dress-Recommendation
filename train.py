import os
import h5py
import numpy as np
import glob
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# Paths
image_directory = 'E:\\D-Project\\fashion_design\\women_fashion\\women fashion\\'
feature_file = 'features.keras'

# Load model
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

# Function to extract features
def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

# Extract features and save if not already done
if not os.path.exists(feature_file):
    image_paths_list = [file for file in glob.glob(os.path.join(image_directory, '*.*'))
                        if file.endswith(('.jpg', '.png', '.jpeg', '.webp'))]
    
    all_features = []
    all_image_names = []

    for img_path in image_paths_list:
        preprocessed_img = preprocess_image(img_path)
        features = extract_features(model, preprocessed_img)
        all_features.append(features)
        all_image_names.append(os.path.basename(img_path))
    
    # Save features and image names
    with h5py.File(feature_file, 'w') as f:
        f.create_dataset('features', data=np.array(all_features))
        f.create_dataset('image_names', data=np.array([name.encode() for name in all_image_names]))
else:
    print("Features already extracted and saved.")
