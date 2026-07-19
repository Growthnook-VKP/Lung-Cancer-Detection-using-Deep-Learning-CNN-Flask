# 🫁 Lung Cancer Detection using Deep Learning (CNN) & Flask

<p align="center">

A Deep Learning-based **Lung Cancer Detection System** developed using **TensorFlow/Keras**, **Flask**, and **Python**. The application classifies lung CT scan images into multiple disease categories using a Convolutional Neural Network (CNN) and provides predictions through both a **Flask Web Application** and a **Tkinter Desktop Application**.

</p>

---

# 📌 Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection significantly improves treatment outcomes. This project leverages **Convolutional Neural Networks (CNNs)** to automatically classify lung CT scan images into six diagnostic categories.

The system includes:

- Deep Learning CNN model
- Flask-based Web Application
- Desktop Prediction Application (Tkinter)
- Automatic Image Preprocessing
- Probability Prediction
- Confidence Score
- Training Accuracy/Loss Visualization

---

# ✨ Features

- 🧠 CNN-based image classification
- 🏥 Six-class lung disease prediction
- 🌐 Flask Web Interface
- 🖥 Desktop GUI using Tkinter
- 📷 CT scan image upload
- 📊 Prediction confidence score
- 📈 Probability distribution visualization
- 🎯 Automatic image preprocessing
- 📉 Early stopping to prevent overfitting
- 💾 Automatic best model saving

---

# 🏥 Disease Categories

The model classifies CT scan images into the following six categories:

- Adenocarcinoma
- Benign
- Large Cell Carcinoma
- Malignant
- Normal
- Squamous Cell Carcinoma

---

# 🛠 Technology Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Framework | Flask |
| GUI | Tkinter |
| Image Processing | Pillow (PIL) |
| Data Handling | NumPy |
| Visualization | Matplotlib |
| Version Control | Git & GitHub |

---

# 📂 Project Structure

```text
Lung-Cancer-Detection/
│
├── app.py                          # Flask Web Application
├── train.py                        # CNN Training Script
├── predict.py                      # Desktop Prediction Script
├── requirements.txt                # Dependencies
├── README.md
│
├── simple_cnn_lung_model.keras     # Generated trained model
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   ├── original_input.png
│   ├── probabilities.png
│   └── model_summary.png
│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/


---

# 🧠 Deep Learning Architecture

The model consists of:

- Input Layer (128 × 128 RGB Image)
- Conv2D (32 Filters)
- MaxPooling2D
- Conv2D (64 Filters)
- MaxPooling2D
- Conv2D (128 Filters)
- MaxPooling2D
- Flatten Layer
- Dense Layer (256 Units)
- Dropout Layer (0.5)
- Softmax Output Layer (6 Classes)

---

# 🔄 Project Workflow

```text
CT Scan Image
       │
       ▼
Image Upload
       │
       ▼
Image Preprocessing
       │
       ▼
Resize (128 × 128)
       │
       ▼
Normalization
       │
       ▼
CNN Model
       │
       ▼
Softmax Prediction
       │
       ▼
Probability Calculation
       │
       ▼
Predicted Disease
       │
       ▼
Confidence Score
```

---

# 📊 Model Training

The CNN model is trained using:

- ImageDataGenerator
- Data Augmentation
- Adam Optimizer
- Categorical Crossentropy Loss
- EarlyStopping
- ModelCheckpoint

Training consists of:

- Training Dataset
- Validation Dataset
- Test Dataset

---

# 📈 Model Performance

The model performance is evaluated using:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

### Training Observation

- Training accuracy increases steadily during training.
- Validation accuracy stabilizes around 70–75%.
- Training loss continuously decreases.
- Validation loss fluctuates slightly, indicating mild overfitting after several epochs.
- EarlyStopping restores the best-performing model weights.

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/Growthnook-VKP/Lung-Cancer-Detection-using-Deep-Learning-CNN-Flask

cd Lung-Cancer-Detection
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📁 Dataset

The dataset used for this project is **not included** in this repository.

It has been excluded because it may be:

- Proprietary
- Confidential
- Too large for GitHub

To train the CNN model, place the dataset in the project directory using the following structure:

```text
dataset/
│
├── train/
├── valid/
└── test/
```

---

# 📦 Trained Model

The trained CNN model (`simple_cnn_lung_model.keras`) is **not included** in this repository.

Generate it by running:

```bash
python train.py
```

This will create:

```text
simple_cnn_lung_model.keras
```

---

# 🚀 Running the Project

## Step 1 – Train the Model

```bash
python train.py
```

This generates:

```text
simple_cnn_lung_model.keras
```

---

## Step 2 – Start the Flask Application

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

---

## Step 3 – Run Desktop Prediction Application

```bash
python predict.py
```

Select a CT scan image and the application will display:

- Predicted Disease
- Prediction Confidence
- Class Probability Distribution

---

# 🖥 Web Application

The Flask application allows users to:

- Upload CT scan images
- Predict lung disease category
- Display confidence score
- Display probability distribution
- Preview uploaded image
- View model architecture summary

---

# 📋 Example Workflow

```text
Upload CT Scan Image
          │
          ▼
Image Preprocessing
          │
          ▼
CNN Prediction
          │
          ▼
Probability Calculation
          │
          ▼
Highest Probability Class
          │
          ▼
Display Prediction & Confidence
```

---

# 🔮 Future Improvements

- Transfer Learning (EfficientNet, ResNet50, DenseNet)
- Explainable AI using Grad-CAM
- Confusion Matrix
- ROC Curve
- Precision-Recall Analysis
- Docker Deployment
- REST API
- Cloud Deployment (AWS / Azure / GCP)
- Mobile Application
- Batch Image Prediction
- Real-time Medical Dashboard

---

# 🤝 Contributing

Contributions are welcome!

1. Fork this repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push your branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Vishal Kumar Prasad**

GitHub: https://github.com/Growthnook-VKP

LinkedIn: www.linkedin.com/in/vishal-kumar-prasad-a1b25b26b

---

# 🌟 Support

If you found this project useful:

⭐ Star this repository

🍴 Fork this repository

💬 Open an Issue for suggestions or improvements

---

<p align="center">
Made with ❤️ using TensorFlow, Keras, Flask and Python
</p>
