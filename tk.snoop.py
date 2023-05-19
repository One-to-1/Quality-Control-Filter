import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib
import customtkinter as ctk
# import ctkinter as ctk
# from customtkinter import messagebox
# from ctkinter import filedialog

global clf
global model_name
global model_name_save

# Load and pre-process image data
def load_data():
    # Load images and labels
    images = []
    labels = []
    for i in range(0, 97):
        image = cv2.imread("obj_img1/img_" + str(i) + ".jpg")
        images.append(image)
        labels.append(1)
    for i in range(0, 99):
        image = cv2.imread("obj_def_img1/img_" + str(i) + ".jpg")
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
# def predict():
#     global clf
#     img_path = filedialog.askopenfilename(initialdir = "/", title = "Select file", filetypes = (("JPG files", "*.jpg"), ("all files", "*.*")))
#     image = cv2.imread(img_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     feature = gray.ravel()
#     prediction = clf.predict([feature])
#     if prediction[0] == 1:
#         good_label.pack(pady=10)
#     else:
#         bad_label.pack(pady=10)
#     show_image(image)

# Live image prediction
def predict():
    global clf
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feature = gray.ravel()
        prediction = clf.predict([feature])
        if prediction[0] == 1:
            cv2.putText(frame, "Good", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Defective", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Output Image", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
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
    # messagebox.showinfo("Accuracy", "The accuracy of the model is: " + str(accrcy))
    welcome.pack_forget()
    load_button.pack_forget()
    save_model_button.pack(pady=10)
    
# Enter Model name for save file
def enter_model_name_save():
    warning_save_model.pack(pady=10)
    model_name_save.pack(pady=10)
    save_model_button2.pack(pady=10)
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
    ctk.CTkLabel(root, text="Model saved successfully!!").pack(pady=10)
    Exit_button.pack(pady=10)
# Enter Model name
def enter_model_name():
    global model_name
    train_button.pack_forget()
    welcome.pack_forget()
    load_button.pack_forget()
    Enter_name_model.pack(pady=10)
    model_name = ctk.CTkEntry(root)
    model_name.pack(pady=10)
    load_button_2.pack(pady=10)
# load model
def load_model(model_name):
    global clf
    clf = joblib.load("saved_models/" + model_name + ".pkl")
    Model_loaded_succ.pack(pady=10)
    load_button_2.pack_forget()
    predict_button.pack(pady=10)

# Exit
def exit():
    root.destroy()

# Frontend
root = ctk.CTk()
root.title("Snoop - Quality Control System")
root.geometry("500x300")

welcome = ctk.CTkLabel(root, text="Hello!\nWelcome to Snoop, Your friendly neighbourhood Quality controll System!!\nWhat Would you Like to Do today?")
welcome.pack(pady=10)

Enter_name_model = ctk.CTkLabel(root, text = "Enter the name of the model: ")

Model_loaded_succ = ctk.CTkLabel(root, text = "Model loaded successfully!!")

good_label = ctk.CTkLabel(root, text="The object is good.")
bad_label = ctk.CTkLabel(root, text="The object is defective.")

warning_save_model = ctk.CTkLabel(root, text="Waring: This may overwrite an existing model!! Make Sure to enter a unique name.")

model_name_save = ctk.CTkEntry(root)

train_button = ctk.CTkButton(root, text="Train a new model", command=train)
train_button.pack(pady=10)

load_button = ctk.CTkButton(root, text="Load a model", command=enter_model_name)
load_button.pack(pady=10)

load_button_2 = ctk.CTkButton(root, text="Load", command=lambda: load_model(model_name.get()))

save_model_button = ctk.CTkButton(root, text="Save Model", command=lambda: enter_model_name_save())

save_model_button2 = ctk.CTkButton(root, text="Save", command=lambda: save_model(model_name_save.get()))

predict_button = ctk.CTkButton(root, text="Predict", command=lambda: predict())

Exit_button = ctk.CTkButton(root, text="Exit", command=exit)
Exit_button.pack(pady=10)

root.mainloop()