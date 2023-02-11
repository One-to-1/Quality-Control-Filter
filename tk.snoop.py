import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib

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

def show_image(image):
    cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Input Image", 360, 640)
    cv2.imshow("Input Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Predict if input image is defective or not
def predict(clf, img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = gray.ravel()
    prediction = clf.predict([feature])
    if prediction[0] == 1:
        return "The object is good."
    else:
        return "The object is defective."

# Train and Save the model
def train():
    images, labels = load_data()
    features = extract_features(images)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print("Accuracy: ", accuracy)
    return clf

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.model_label = tk.Label(self, text="Model: ")
        self.model_label.pack(side="top")

        self.model_entry = tk.Entry(self)
        self.model_entry.pack(side="top")

        self.image_label = tk.Label(self, text="Image: ")
        self.image_label.pack(side="top")

        self.image_entry = tk.Entry(self)
        self.image_entry.pack(side="top")

        self.predict_button = tk.Button(self)
        self.predict_button["text"] = "Predict"
        self.predict_button["command"] = self.predict
        self.predict_button.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
        self.quit.pack(side="bottom")

    def predict(self):
        model_name = self.model_entry.get()
        image_path = self.image_entry.get()
        clf = joblib.load("saved_models/" + model_name + ".pkl")
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = gray.ravel()
        prediction = clf.predict([feature])
        if prediction[0] == 1:
            tk.messagebox.showinfo("Result", "The object is good.")
        else:
            tk.messagebox.showinfo("Result", "The object is defective.")
        show_image(image)

root = tk.Tk()
app = Application(master=root)
app.mainloop()
