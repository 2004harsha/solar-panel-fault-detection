# Solar Panel Fault Detection using CNN

This project is focused on classifying various types of faults in solar panels using a Convolutional Neural Network (CNN). The goal is to automate the detection of common issues such as dust, physical damage, and electrical damage that impact the efficiency of solar panels.

---

## Project Overview

* **Objective**: Develop a deep learning model that classifies solar panel images into specific fault categories.
* **Methodology**: A custom CNN is trained from scratch to recognize visual patterns associated with different fault types.
* **Platform**: Developed and tested in Google Colab using TensorFlow/Keras.

---

## Dataset Details

The dataset contains labeled images of solar panels across six categories:

* **Bird-drop**
* **Clean**
* **Dusty**
* **Electrical-damage**
* **Physical-Damage**
* **Snow-Covered**

### Folder Structure

```
dataset/
├── train/
│   ├── Bird-drop/
│   ├── Clean/
│   ├── Dusty/
│   ├── Electrical-damage/
│   ├── Physical-Damage/
│   └── Snow-Covered/
└── test/
    ├── Bird-drop/
    ├── Clean/
    ├── Dusty/
    ├── Electrical-damage/
    ├── Physical-Damage/
    └── Snow-Covered/
```

*Note: The dataset is not included in the GitHub repository due to size limitations. It is loaded from Google Drive within the notebook.*

---

## Model Architecture

A Convolutional Neural Network (CNN) is used with the following architecture:

* **Input Shape**: 224 x 224 x 3 RGB images
* **Conv2D + MaxPooling + Dropout** layers (3 blocks)
* **Flatten**
* **Dense Layer (ReLU)**
* **Output Layer**: 6 neurons with Softmax activation (for 6 classes)

The model is compiled with:

* **Loss Function**: `categorical_crossentropy`
* **Optimizer**: `adam`
* **Metrics**: `accuracy`

---

## Training Summary

* **Training Accuracy**: \~99%
* **Validation Accuracy**: \~92%
* **Smooth training convergence**, minimal overfitting

Training and validation graphs (accuracy and loss) are included in the notebook.

---

## Evaluation and Results

The notebook includes:

* **Classification Report**: Precision, Recall, F1-score for each class
* **Confusion Matrix**
* **Visual Predictions**: Random test images with predicted and true labels

The model performs well across all fault categories, demonstrating strong generalization on unseen data.

---

## How to Run

1. **Clone the repository:**

   ```bash
   git clone https://github.com/2004harsha/solar-panel-fault-detection.git
   cd solar-panel-fault-detection
   ```

2. **Open the notebook:**

   * `solar_fault_detection.ipynb`

3. **In Google Colab:**

   * Mount your Google Drive to access the dataset:

     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

4. **Install required packages (if not already installed):**

   ```bash
   pip install tensorflow opencv-python
   ```

5. **Run all cells** in the notebook. Ensure the dataset path is correct and matches the expected structure.

---

## Future Improvements

* Add **data augmentation** for more robustness
* Implement **transfer learning** using pre-trained CNNs like ResNet or EfficientNet
* Build a **frontend dashboard** for deployment (Streamlit/Flask)
* Train on more **real-world and larger datasets**

---

## Author

**Harsha Purohit**
GitHub: [@2004harsha](https://github.com/2004harsha)

---

## Disclaimer

This project is intended for academic and demonstration purposes. The model was trained on a specific dataset, and performance may vary with other data or real-world conditions.
