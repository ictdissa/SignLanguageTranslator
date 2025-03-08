#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Dataset location
DATA_DIR = "CollectedGestures"
gestures = np.array(["hello", "thanks", "yes", "no", "please"])

# Load dataset
samples, labels = [], []
for gesture_idx, gesture in enumerate(gestures):
    for file in os.listdir(os.path.join(DATA_DIR, gesture)):
        sample = np.load(os.path.join(DATA_DIR, gesture, file))
        samples.append(sample)
        labels.append(gesture_idx)

X = np.array(samples)
y = to_categorical(labels, num_classes=len(gestures))

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(20, 258)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(gestures), activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_val, y_val))

# Save the trained model
model.save("gesture_recognition_model.h5")


# In[ ]:




