#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import cv2
import numpy as np
import mediapipe as mp
import pyperclip
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("gesture_recognition_model.h5")

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define gesture labels
actions = np.array(['hello', 'bye', 'yes', 'no', 'please'])

# Process image using MediaPipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 4))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()])

# Draw hand landmarks
def draw_styled_landmarks(img, results):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=5),
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=5),
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3))

# Initialize webcam
WIDTH, HEIGHT = 640, 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

sequence, predictions, sentence, confirmed = [], [], [], []
threshold, counter = 0.5, 0

# Start real-time gesture recognition
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame through MediaPipe
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # Extract keypoints and store the last 20 frames
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-20:]

        # Make a prediction when we have enough frames
        if len(sequence) == 20:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            # Confirm gesture if consistently predicted
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) == 0:
                        sentence.append(actions[np.argmax(res)])
                    else:
                        sentence[-1] = actions[np.argmax(res)]
            
            # Save confirmed phrase when 'Enter' key is pressed
            if cv2.waitKey(1) == 13:
                sentence.append(actions[np.argmax(res)])
                confirmed = sentence[:-1]
                counter += 1

            # Keep only the last 10 recognized words
            if len(sentence) > 10:
                sentence = sentence[-10:]

            # Display predictions on the frame
            cv2.rectangle(image, (0,0), (WIDTH, 50), (50, 50, 255), -1)
            cv2.putText(image, ' '.join(sentence), (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, ' '.join(confirmed), (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the updated frame
        cv2.imshow('Gesture Recognition', image)

        # Exit and control options
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('c'):
            pyperclip.copy(' '.join(sentence))
        elif key & 0xFF == ord('s'):
            if len(sentence) != 0:
                with open("signed_text.txt", "a") as f:
                    f.write(' '.join(sentence[:-1]) + "\n")

cap.release()
cv2.destroyAllWindows()


# In[ ]:




