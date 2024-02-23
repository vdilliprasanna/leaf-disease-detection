# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 09:40:45 2023

@author: Dilli Prasanna
"""


import os
import numpy as np
import cv2
data_dir = 'D:/winsem-22-23/imva/pla/PlantVillage/'
categories = ['Pepper__bell___healthy','Pepper__bell___Bacterial_spot']
labels = [i for i in range(len(categories))]

img_size = 224

dataset = []

for category in categories:
    folder_path = os.path.join(data_dir, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            # convert image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # resize the image to the required size for the CNN
            resized = cv2.resize(gray, (img_size, img_size))
            dataset.append([resized, labels[categories.index(category)]])
        except Exception as e:
            print(e)

from sklearn.model_selection import train_test_split

X = []
y = []

for features, label in dataset:
    X.append(features)
    y.append(label)

X = np.array(X) / 255.0
X = np.reshape(X, (X.shape[0], img_size, img_size, 1))

y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svm = SVC(kernel='linear', random_state=42)

svm.fit(X_train.reshape(X_train.shape[0], -1), y_train)

svm_predictions = svm.predict(X_test.reshape(X_test.shape[0], -1))

svm_accuracy = accuracy_score(y_test, svm_predictions)

print("SVM accuracy: ", svm_accuracy)

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(categories))
])

cnn.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

cnn_accuracy = cnn.evaluate(X_test, y_test)[1]

print("CNN accuracy: ", cnn_accuracy)

def predict(image_path):
    img = cv2.imread(image_path)

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (img_size, img_size))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, img_size, img_size, 1))
        cnn_prediction = cnn.predict(reshaped)
        svm_prediction = svm.predict(reshaped.reshape(1, -1))
        cnn_score = tf.nn.softmax(cnn_prediction[0])
        svm_score = svm.decision_function(reshaped.reshape(1, -1))

        print("CNN prediction:", categories[np.argmax(cnn_score)])
        print("SVM prediction:", categories[np.argmax(svm_score)])  
    except Exception as e:
        print(e)
predict('C:/Users/Dilli Prasanna/Downloads/archive/PlantVillage/Pepper__bell___Bacterial_spot/0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG')
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# SVM confusion matrix
svm_cm = confusion_matrix(y_test, svm_predictions)
sns.heatmap(svm_cm, annot=True, cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("SVM Confusion Matrix")
plt.show()

# CNN confusion matrix
cnn_predictions = np.argmax(cnn.predict(X_test), axis=1)
cnn_cm = confusion_matrix(y_test, cnn_predictions)
sns.heatmap(cnn_cm, annot=True, cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("CNN Confusion Matrix")
plt.show()

# Comparison bar chart
models = ["SVM", "CNN"]
accuracies = [svm_accuracy, cnn_accuracy]
plt.bar(models, accuracies)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Comparison of SVM and CNN Accuracy")
plt.show()
from sklearn.metrics import classification_report
svm_report = classification_report(y_test, svm_predictions)
cnn_predictions = np.argmax(cnn.predict(X_test), axis=-1)
cnn_report = classification_report(y_test, cnn_predictions)

print("SVM Report:\n", svm_report)
print("CNN Report:\n", cnn_report)
