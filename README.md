# Solar Panel Fault Detection using CNN

This project implements a Convolutional Neural Network (CNN) to automatically detect faults in solar panels from images. The model classifies images into six fault categories to aid in predictive maintenance and improve solar energy efficiency.

## Project Overview

* **Type:** Image Classification using CNN
* **Model:** Custom-built CNN in TensorFlow/Keras
* **Classes:**

  * Bird-drop
  * Clean
  * Dusty
  * Electrical-damage
  * Physical-Damage
  * Snow-Covered

## Directory Structure

```
solar-panel-fault-detection/
├── cnn.ipynb                # Jupyter Notebook with full implementation
├── README.md                # Project overview and instructions
└── (dataset not included due to size)
```

## Key Features

* Preprocessing using `ImageDataGenerator` with rescaling and augmentation
* CNN architecture with Conv2D, MaxPooling, Dropout, and Dense layers
* Model trained with Categorical Crossentropy loss and Adam optimizer
* Evaluation metrics include accuracy and loss plots

## Model Architecture

```
Input Layer (150x150x3)
→ Conv2D → ReLU → MaxPooling
→ Conv2D → ReLU → MaxPooling
→ Flatten → Dense → Dropout
→ Dense → Output Layer (Softmax)
```

## Sample Code Snippet

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])
```

## Dataset

* **Format:** Directory-structured image dataset
* **Location:** Not uploaded to GitHub (available via request or external link)
* **Image Size:** 150x150 RGB

## Results

* Achieved high accuracy on validation and test sets
* Model effectively distinguishes between various types of solar panel faults

## Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/2004harsha/solar-panel-fault-detection.git
   cd solar-panel-fault-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt  # if requirements file is added
   # or manually install
   pip install tensorflow matplotlib numpy
   ```

3. Run the notebook:

   ```bash
   jupyter notebook cnn.ipynb
   ```

## Future Work

* Deploy as a web app using Flask or Streamlit
* Automate fault alerts with confidence thresholds

## License

This project is for educational use. All rights belong to the respective dataset creators.

---

**Author:** Harsha Purohit
**GitHub:** [@2004harsha](https://github.com/2004harsha)
