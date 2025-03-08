# 🚀 Gesture Recognition with MediaPipe & LSTM
This project enables real-time hand gesture recognition using MediaPipe for landmark detection and LSTM (Long Short-Term Memory) neural networks for classification.

# 📌 Features
✅ Real-time hand tracking using MediaPipe
✅ Gesture dataset collection for training
✅ Deep learning classification with LSTM networks
✅ Live recognition and prediction of gestures
✅ Copy or save recognized gestures as text

# 🛠️ Technology Stack & Frameworks
This project leverages computer vision and deep learning frameworks:

Python	Primary programming language.
OpenCV	Handles video capture and frame processing.
MediaPipe	Detects hands, pose, and key landmarks.
TensorFlow/Keras	Trains an LSTM model for gesture classification.
NumPy	Handles numerical data processing.
Pyperclip	Copies recognized gestures to clipboard.

# 📂 Repository Structure
📁 GestureRecognitionProject/
│── 📁 data/                # (Optional) Pre-collected datasets
│── 📁 models/              # (Optional) Trained models
│── CollectDataset.py       # Script for dataset collection
│── TrainData.py            # Script for model training
│── GestureRecognition.py   # Real-time gesture recognition
│── requirements.txt        # Dependencies for the project
│── README.md               # Project overview & instructions
│── .gitignore              # Ignore unnecessary files (optional)

# 🛠️ Setup & Installation
To run this project on your local machine, follow these steps:

1️⃣ Install Dependencies
Run the following command to install all required packages:
pip install -r requirements.txt
2️⃣ Collect Gesture Data
To create a dataset of gestures, run the dataset collection script:
python CollectDataset.py
📌 This script will capture hand movements and save them as numerical keypoints.
3️⃣ Train the LSTM Model
Once the dataset is collected, train the gesture recognition model:
python TrainData.py
📌 This script will train an LSTM deep learning model using collected keypoints.
4️⃣ Run Gesture Recognition
After training, run the real-time gesture recognition:
python GestureRecognition.py
📌 This script detects gestures live, classifies them, and displays predictions on the screen.

🔬 How Gesture Recognition Works
This project follows a 3-step process:
1️⃣ Hand & Pose Tracking with MediaPipe
MediaPipe Holistic detects hands, pose, and facial landmarks.
It extracts 21 keypoints per hand, returning their x, y, and z coordinates.
These keypoints act as input features for the deep learning model.
2️⃣ Training an LSTM Deep Learning Model
The collected keypoints are converted into sequences of 20 frames.
These sequences are used to train an LSTM network that learns gesture patterns over time.
The trained model outputs a probability distribution over different gestures.
3️⃣ Real-Time Gesture Classification
OpenCV captures video frames.
MediaPipe extracts keypoints.
The trained LSTM model classifies the gesture in real time.
The predicted gesture appears on the screen and can be copied or saved.
🎯 Future Improvements
🔹 Expand gesture vocabulary by collecting more training data.
🔹 Optimize LSTM model for lower latency in real-time recognition.
🔹 Deploy as a web-based app using Flask or Streamlit.
🔹 Enhance accuracy by implementing data augmentation techniques.


Fork the repository.
Create a feature branch.
Submit a pull request.
For major changes, open an issue first to discuss improvement
