# program to create a dataset of images and labels by taking live input from webcam and saving it to a folder

import cv2
import os

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_image(image, label, count, directory):
    file_name = f"{label}_{count}.jpg"
    file_path = os.path.join(directory, file_name)
    cv2.imwrite(file_path, image)
    return file_path

def main():
    label = "img"
    directory = "./obj_def_img"
    create_directory(directory)

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    count = 0

    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            file_path = save_image(frame, label, count, directory)
            print(f"Saved image: {file_path}")
            count += 1
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
