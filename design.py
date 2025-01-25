import os
import h5py
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine
from PIL import Image
import matplotlib.pyplot as plt
import flet as ft

# Initialize model and load features
feature_file = 'features.keras'
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

# Load precomputed features
with h5py.File(feature_file, 'r') as f:
    all_features = np.array(f['features'])
    all_image_names = [name.decode() for name in f['image_names']]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

def recommend_fashion_items(input_image_path, top_n=5):
    preprocessed_img = preprocess_image(input_image_path)
    input_features = extract_features(model, preprocessed_img)

    similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
    similar_indices = np.argsort(similarities)[-top_n:]
    similar_indices = [idx for idx in similar_indices if all_image_names[idx] != os.path.basename(input_image_path)]
    
    # Display recommendations
    plt.figure(figsize=(15, 10))
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(Image.open(input_image_path))
    plt.title("Input Image")
    plt.axis('off')

    for i, idx in enumerate(similar_indices[:top_n], start=1):
        image_path = os.path.join('E:\\D-Project\\fashion_design\\women_fashion\\women fashion\\', all_image_names[idx])
        plt.subplot(1, top_n + 1, i + 1)
        plt.imshow(Image.open(image_path))
        plt.title(f"Recommendation {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def main(page: ft.Page):
    page.window.width = 400
    page.window.height = 800
    page.padding = 0
    page.theme_mode = ft.ThemeMode.LIGHT
    page.fonts = {
        "Poppinsbold": "/fonts/Poppins-Bold.ttf",
        "Poppins": "/fonts/Poppins-Medium.ttf"
    }
    page.theme = ft.Theme(font_family="Poppins")

    def route_change(route):
        page.views.clear()
        page.theme_mode = ft.ThemeMode.LIGHT
        page.theme = ft.theme.Theme(color_scheme_seed='#A233A2')
        page.views.append(ft.View("/", [splash_screen], padding=0))

        if page.route == "/home":
            page.views.append(
                ft.View(
                    "/home",
                    [
                        ft.Text(value="Upload an Image", font_family="Poppins", size=20),
                        upload_button,
                        recommendation_image
                    ],
                    padding=0,
                    vertical_alignment="center",
                    horizontal_alignment="center",
                )
            )
        page.update()

    def open_upload_dialog(e):
        upload_dialog.open = True
        page.update()

    def submit_image_path(e):
        input_image_path = path_input.value
        recommendation_image.src = input_image_path
        recommendation_image.update()
        upload_dialog.open = False
        recommend_fashion_items(input_image_path, top_n=4)

    upload_button = ft.ElevatedButton("Upload Image", on_click=open_upload_dialog)
    path_input = ft.TextField(label="Enter image path", width=300)
    submit_button = ft.ElevatedButton("Submit", on_click=submit_image_path)

    upload_dialog = ft.AlertDialog(
        title=ft.Text("Upload Image"),
        content=path_input,
        actions=[submit_button],
        actions_alignment="end",
    )

    recommendation_image = ft.Image(width=300, height=300)

    splash_screen_data = ft.Column(
        [
            ft.Container(height=50),
            ft.Text(value="FASHION DESIGN", font_family="Poppins", size=30),
            ft.Image(src="/images/logo.png", width=300),
            ft.Container(height=30),
            ft.ElevatedButton("START", on_click=lambda _: page.go("/home"))
        ],
        horizontal_alignment='center'
    )

    splash_screen = ft.Container(
        content=splash_screen_data,
        gradient=ft.LinearGradient(
            begin=ft.alignment.top_center,
            end=ft.alignment.bottom_center,
            colors=('#E4EEE9', '#93A5CE')
        ),
        width=400,
        height=800,
        padding=25
    )

    def view_pop(view):
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    page.dialog = upload_dialog
    page.on_route_change = route_change
    page.on_view_pop = view_pop
    page.go(page.route)

ft.app(target=main, assets_dir="assets")
