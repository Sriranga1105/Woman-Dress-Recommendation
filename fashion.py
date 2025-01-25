import os
import h5py
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine
from PIL import Image
import matplotlib.pyplot as plt
import glob 
# Paths
image_directory = 'E:\\D-Project\\fashion_design\\women_fashion\\women fashion\\'
feature_file = 'features.h5'

# Load or create model
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

# Check if features file exists, load if so
if os.path.exists(feature_file):
    with h5py.File(feature_file, 'r') as f:
        all_features = np.array(f['features'])
        all_image_names = [name.decode() for name in f['image_names']]
else:
    # Otherwise, extract features and save
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

# Recommendation function remains the same
def recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model, top_n=5):
    preprocessed_img = preprocess_image(input_image_path)
    input_features = extract_features(model, preprocessed_img)

    similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
    similar_indices = np.argsort(similarities)[-top_n:]

    similar_indices = [idx for idx in similar_indices if all_image_names[idx] != os.path.basename(input_image_path)]

    plt.figure(figsize=(15, 10))
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(Image.open(input_image_path))
    plt.title("Input Image")
    plt.axis('off')

    for i, idx in enumerate(similar_indices[:top_n], start=1):
        image_path = os.path.join(image_directory, all_image_names[idx])
        plt.subplot(1, top_n + 1, i + 1)
        plt.imshow(Image.open(image_path))
        plt.title(f"Recommendation {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Test with an input image
input_image_path = 'E:\\D-Project\\fashion_design\\women_fashion\\women fashion\\black floral saree.jpg'  # Replace with your input image path
recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model, top_n=4)
