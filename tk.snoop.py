import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog

global clf
global model_name
global model_name_save

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
def predict():
    global clf
    img_path = filedialog.askopenfilename(initialdir = "/", title = "Select file", filetypes = (("JPG files", "*.jpg"), ("all files", "*.*")))
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = gray.ravel()
    prediction = clf.predict([feature])
    if prediction[0] == 1:
        good_label.pack()
    else:
        bad_label.pack()
    show_image(image)
# Train Model
def train():
    global accrcy
    global clf
    images, labels = load_data()
    features = extract_features(images)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)
    accrcy = clf.score(X_test, y_test)
    messagebox.showinfo("Accuracy", "The accuracy of the model is: " + str(accrcy))
    welcome.pack_forget()
    load_button.pack_forget()
    save_model_button.pack()
    
# Enter Model name for save file
def enter_model_name_save():
    warning_save_model.pack()
    model_name_save.pack()
    save_model_button2.pack()
# Save model
def save_model(name):
    global clf
    joblib.dump(clf, "saved_models/" + name + ".pkl")
    train_button.pack_forget()
    Exit_button.pack_forget()
    save_model_button.pack_forget()
    warning_save_model.pack_forget()
    model_name_save.pack_forget()
    save_model_button2.pack_forget()
    tk.Label(root, text="Model saved successfully!!").pack()
    Exit_button.pack()
# Enter Model name
def enter_model_name():
    global model_name
    train_button.pack_forget()
    welcome.pack_forget()
    load_button.pack_forget()
    Enter_name_model.pack()
    model_name = tk.Entry(root)
    model_name.pack()
    load_button_2.pack()
# load model
def load_model(model_name):
    global clf
    clf = joblib.load("saved_models/" + model_name + ".pkl")
    Model_loaded_succ.pack()
    load_button_2.pack_forget()
    predict_button.pack()

# Exit
def exit():
    root.destroy()

# Frontend
root = tk.Tk()
root.title("Snoop - Quality Control System")
root.geometry("500x300")

welcome = tk.Label(root, text="Hello!\nWelcome to Snoop, Your friendly neighbourhood Quality controll System!!\nWhat Would you Like to Do today?")
welcome.pack()

Enter_name_model = tk.Label(root, text = "Enter the name of the model: ")

Model_loaded_succ = tk.Label(root, text = "Model loaded successfully!!")

good_label = tk.Label(root, text="The object is good.")
bad_label = tk.Label(root, text="The object is defective.")

warning_save_model = tk.Label(root, text="Waring: This may overwrite an existing model!! Make Sure to enter a unique name.")

model_name_save = tk.Entry(root)

train_button = tk.Button(root, text="Train a new model", command=train)
train_button.pack()

load_button = tk.Button(root, text="Load a model", command=enter_model_name)
load_button.pack()

load_button_2 = tk.Button(root, text="Load", command=lambda: load_model(model_name.get()))

save_model_button = tk.Button(root, text="Save Model", command=lambda: enter_model_name_save())

save_model_button2 = tk.Button(root, text="Save", command=lambda: save_model(model_name_save.get()))

predict_button = tk.Button(root, text="Predict", command=lambda: predict())

Exit_button = tk.Button(root, text="Exit", command=exit)
Exit_button.pack()

root.mainloop()