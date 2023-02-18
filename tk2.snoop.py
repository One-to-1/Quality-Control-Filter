import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib
import tkinter as tk
from tkinter import messagebox

class detector:
    def __init__(self, master):
        self.master = master
        self.pages = []
        self.clf = None
        
        self.create_widgets()
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
    def predict(clf):
        img_path = input("Enter the path of the image: ")
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = gray.ravel()
        prediction = clf.predict([feature])
        if prediction[0] == 1:
            print("The object is good.")
        else:
            print("The object is defective.")
        show_image(image)
    # Train Model
    def train():
        images, labels = load_data()
        features = extract_features(images)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print("Accuracy: ", accuracy)
        return clf
    # Save model
    def save_model(clf):
        print("Waring: This may overwrite an existing model!! Make Sure to enter a unique name.")
        name = input("Enter the name of the model: ")
        joblib.dump(clf, "saved_models/" + name + ".pkl")
        print("Model saved successfully as " + name + ".pkl")
        return clf
    # load model
    def load_model():
        name = input("Enter the name of the model: ")
        clf = joblib.load("saved_models/" + name + ".pkl")
        return clf
    # Create widgets
    def create_widgets(self):
        # Create the main menu
        main_menu = tk.Frame(self.master)
        tk.Label(main_menu, text="Snoop - Quality Control System", font=("Calibri", 20)).pack()
        tk.Button(main_menu, text="Train Model", command=self.train).pack()
        tk.Button(main_menu, text="Load Model", command=self.load_model).pack()
        self.pages.append(main_menu)
        
        # Create Train Page
        trainpage = tk.Frame(self.master)
        
        
        self.show_main_menu()
        
    def show_main_menu(self):
        self.pages[0].pack()
        # self.pages[1].pack_forget()
        # self.pages[1].pack_forget()
        # self.pages[1].pack_forget()
        # self.pages[1].pack_forget()
        # self.pages[1].pack_forget()
               

root = tk.Tk()
root.title("Snoop - Quality Control System")
root.geometry("500x500")
detector(root)
root.mainloop()