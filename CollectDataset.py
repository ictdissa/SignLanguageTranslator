#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe models
mp_holistic = mp.solutions.holistic  
mp_drawing = mp.solutions.drawing_utils  

# Define gestures
gestures = ["hello", "thanks", "yes", "no", "please"]
DATA_DIR = "CollectedGestures"

# Number of sequences per gesture
samples_per_gesture = 50  
frames_per_sample = 20  

# Create dataset folders
for gesture in gestures:
    gesture_path = os.path.join(DATA_DIR, gesture)
    os.makedirs(gesture_path, exist_ok=True)

# Function to extract keypoints
def get_landmark_features(results):
    pose = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    left_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    right_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, left_hand, right_hand])

# Start video capture
video = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for gesture in gestures:
        for sample in range(samples_per_gesture):
            sequence = []

            print(f"Recording {gesture.upper()} ({sample + 1}/{samples_per_gesture})")
            time.sleep(1)  # Small delay to prepare

            for frame in range(frames_per_sample):
                ret, frame_data = video.read()
                frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                
                keypoints = get_landmark_features(results)
                sequence.append(keypoints)

                # Draw hand and body landmarks
                mp_drawing.draw_landmarks(frame_data, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(frame_data, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(frame_data, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # Display recording info
                cv2.putText(frame_data, f"Recording {gesture.upper()} - {sample + 1}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Dataset Recorder', frame_data)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Save collected sequence
            np.save(os.path.join(DATA_DIR, gesture, f"sample_{sample}.npy"), np.array(sequence))

video.release()
cv2.destroyAllWindows()


# In[ ]:




