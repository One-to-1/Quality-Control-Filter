import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
# Load and pre-process image data
def load_data():
    # Load images and labels
    images = []
    labels = []
    for i in range(1, 25):
        image = cv2.imread("obj(" + str(i) + ").jpg")
        images.append(image)
        labels.append(1)
    for i in range(1, 11):
        image = cv2.imread("def_obj(" + str(i) + ").jpg")
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
# Train and test SVM model
def train_test_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print("Accuracy: ", accuracy)
# Predict if input image is defective or not
def predict(image, clf):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = gray.ravel()
    prediction = clf.predict([feature])
    if prediction[0] == 1:
        print("The object is good.")
    else:
        print("The object is defective.")
# Main function
if __name__ == '__main__':
    images, labels = load_data()
    features = extract_features(images)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print("Accuracy: ", accuracy)
    # Take input image - Ask for input image path
    img_path = input("Enter the path of the image: ")
    input_image = cv2.imread(img_path)
    predict(input_image, clf)
    cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Input Image", 360, 640)
    cv2.imshow("Input Image", input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()