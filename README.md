# ✋ Vision-Based Hand Gesture Recognition

A real-time hand gesture recognition system built using Python, MediaPipe, OpenCV, and a Random Forest machine learning classifier. The project detects hand landmarks through a webcam and recognizes gestures to support touchless computer interaction.

## 🚀 Features

- Real-time hand detection using a webcam
- Recognition of six hand gestures
- 21 hand landmarks extracted using MediaPipe
- 42 normalized features for gesture classification
- Random Forest-based gesture recognition
- Modular architecture for gesture detection and action execution
- Dataset collection, validation, model training, and evaluation scripts
- Gesture-based touchless interaction

## 🖐️ Supported Gestures

- Open Palm
- Fist
- Peace
- Thumbs Up
- Thumbs Down
- Pointing

## 🛠️ Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV
- **Hand Landmark Detection:** MediaPipe
- **Machine Learning:** Scikit-learn — Random Forest
- **Data Processing:** NumPy, Pandas

## 📁 Project Structure

```text
Hand-gesture/
├── actions/          # Gesture-triggered actions
├── core/             # Detection, features, intent and activation logic
├── dataset/          # Gesture dataset
├── models/           # Trained classification model
├── app.py            # Main application
├── camera_test.py    # Webcam testing
├── collect_data.py   # Dataset collection
├── detector.py       # Gesture detection
├── train_model.py    # Model training
├── hand_landmarker.task
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/mitsocswaroopjadhav-star/hand-gesture-recognition-.git
cd hand-gesture-recognition-
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Ensure that your webcam is connected and that the required model files are present.

## 🧠 Machine Learning Approach

The system uses MediaPipe to detect 21 hand landmarks. These landmarks are converted into 42 normalized features, which are passed to a trained Random Forest classifier to predict the gesture.

The model can be evaluated using the project's evaluation scripts.

## 📊 Model Evaluation

The reported test accuracy during development was **98.76%** on a held-out test set of 241 samples. Results may vary with new users, lighting conditions, camera angles, and real-world usage.

## 🔮 Future Improvements

- Improve robustness under different lighting conditions
- Add face recognition and user-specific profiles
- Integrate natural-language or LLM-based commands
- Package the application for easier installation
- Evaluate performance across more users and environments

## 👨‍💻 Author

**Swaroop Jadhav**

Developed as a computer vision and machine learning project.
