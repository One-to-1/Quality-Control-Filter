import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib
global clf
# Load and pre-process image data
def load_data():
    # Load images and labels
    images = []
    labels = []
    for i in range(1, 25):
        image = cv2.imread("obj_img/obj(" + str(i) + ").jpg")
        images.append(image)
        labels.append(1)
    for i in range(1, 11):
        image = cv2.imread("obj_def_img/def_obj(" + str(i) + ").jpg")
        images.append(image)
        labels.append(0)
    return images, labels

# Extract features from image using OpenCV
def extract_features(images):
    features = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features.append(gray.ravel())
    return features

# Predict if input image is defective or not
def predict(clf, image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = gray.ravel()
    prediction = clf.predict([feature])
    if prediction[0] == 1:
        messagebox.showinfo("Result", "The object is good.")
    else:
        messagebox.showinfo("Result", "The object is defective.")

# Train and Save the model
def train():
    images, labels = load_data()
    features = extract_features(images)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
    # global clf
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    messagebox.showinfo("Accuracy", "Accuracy: " + str(accuracy))
    # return clf

def select_image():
    root.filename = filedialog.askopenfilename(initialdir = "/", title = "Select file", filetypes = (("JPG files", "*.jpg"), ("all files", "*.*")))
    predict(clf, root.filename)

root = tk.Tk()
root.title("Snoop - Quality Control System")
root.geometry("500x500")

train_button = tk.Button(root, text="Train a new model", command=train)
train_button.pack()

select_image_button = tk.Button(root, text="Select Image", command=select_image)
select_image_button.pack()

use_existing_model_button = tk.Button(root, text="Use Existing Model", command=select_image)
use_existing_model_button.pack()

root.mainloop()
