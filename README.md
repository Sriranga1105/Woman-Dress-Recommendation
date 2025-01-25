# Woman-Dress-Recommendation #
Welcome to the Woman Fashion Recommendation project! This is a simple yet powerful tool to help you find fashion inspirations based on an image you upload. Think of it as your personal AI stylist.

## 🎯 Project Overview ##
This project uses deep learning to analyze fashion images and suggest visually similar outfits. Whether it's a traditional saree or a chic modern dress, the system recommends items that match the uploaded image.

We use the VGG16 neural network, a pre-trained deep learning model known for its excellent performance in image recognition tasks, to extract features from images and find similarities.

## ✨ Final Outcome of the project ##
https://github.com/user-attachments/assets/db04ae2d-07da-4816-ba42-3d7dae877930

*After the submit it will give an output like this,👇*
![Figure_1](https://github.com/user-attachments/assets/cd2b6879-ac79-40fe-8a2a-38147ba9717d)

## 🛠 How It Works ##
1. **Preprocessing:** The input image is resized to fit the model's requirements.
 
2. **Feature Extraction:** We leverage the VGG16 model (without the top classification layer) to get a vector representation of the image.
3. **Similarity Matching:** The system calculates cosine similarity to compare the input image with a database of precomputed fashion item features.
4. **Recommendations:** The top 4 visually similar items are displayed.

## 🚀 Features ##
* **Interactive UI:** Upload your image through a clean and simple interface.

* **Real-Time Suggestions:** Get recommendations in seconds.
* **Precomputed Data:** Faster results thanks to pre-extracted features from a large dataset of women’s fashion images.

## 📂 Code Breakdown ##
### Key Files ###
* **```design.py:```** Contains the main app logic and UI implementation using the Flet framework.

* **```fashion.py:```** Handles feature extraction and recommendations.
* **```train.py:```** Precomputes features for the image dataset.

### Why VGG16? ###
We chose VGG16 because:
* It’s lightweight and well-suited for transfer learning tasks.

* Its architecture works great for extracting meaningful image features.

## 💻 System Requirement ##
* Operating System: Windows 10 or Above

* IDE: PyCharm or VS Code
* Programming Language: Python (use current updated version)
* Python libraries: Flet (use current updated version)

## 🔧 How to Run ##
1. Clone the repository:
```bash
git clone https://github.com/Sriranga1105/Woman-Dress-Recommendation.git
cd woman-fashion-recommendation
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```
3. Add your fashion dataset in the specified folder path.

4. Precompute features (if not already saved):
```bash
python train.py
```

5. Launch the app:
```bash
python design.py
```

## 🌟 What Makes It Special? ##
It’s personal! You can upload your own photos and explore fashion items that suit your style. The AI doesn't just suggest random options it finds items with a similar aesthetic.

## 🤝 Contributing ##
Got ideas to make it better? Feel free to fork the repo, make your changes, and submit a pull request.

## 📬 Feedback ##
I love to hear your thoughts. If you’ve got feedback, reach out via email or open an issue on GitHub.
