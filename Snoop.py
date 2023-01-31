import cv2
import numpy as np
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
# Train and Save the model
def train():
    train = 'y'
    while train == 'y':
        images, labels = load_data()
        features = extract_features(images)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print("Accuracy: ", accuracy)
        save = input("Do you want to save the model? [y/n]: ")
        if save == 'y':
            print("Waring: This may overwrite an existing model!! Make Sure to enter a unique name.")
            name = input("Enter the name of the model: ")
            joblib.dump(clf, "saved_models/" + name + ".pkl")
            print("Model saved successfully as " + name + ".pkl")
            return clf
        if save == 'n':
            train = input("Would you like to train the model again? [y/n]: ")
            if train == 'n':
                return clf
def menu_opt():
    print("1. Train a new model\n2. Use an existing model\n0. Exit\n")
    opt = int(input("Enter your option: "))
    return opt
print("Hello!\nWelcome to Snoop, Your friendly neighbourhood Quality controll System!!\nWhat Would you Like to Do today?\n")
sel = menu_opt()
while sel != 0:
    if sel == 1:
        print("Training model...\n")
        clf = train()
        predict(clf)
        print("Training complete.\n")
        sel = menu_opt()
    if sel == 2:
        name = input("Enter the name of the model: ")
        clf = joblib.load("saved_models/" + name + ".pkl")
        predict(clf)
        sel = menu_opt()