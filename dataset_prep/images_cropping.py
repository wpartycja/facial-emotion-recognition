from pathlib import Path
import numpy as np
from IPython.display import Image, display
import time
import cv2
import dlib
import os

output = '/faces/output.png'

# @TODO: dokonczyc typehinty

def show_img(img):
    cv2.imwrite(output, img)
    display(Image(filename=output, width=800, height=600))
    return

def load_img(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    loads image and changes channel colour
    """
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, rgb


def convert_and_trim_bb(image: np.ndarray, rect):
    """
    reads bounding box corner coordinates
    and converts it to coordinates of left upper corner
    and width and height
    """
    
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    
    return (startX, startY, w, h)


def cnn_crop(path, save_path):
    """
    detects face from image using cnn algorithm
    """
    img, rgb = load_img(path)
    cnn_detector = dlib.cnn_face_detection_model_v1("./input/mmod_human_face_detector.dat")
    cnn_rects = cnn_detector(rgb, 1)

    cnn_boxes = [convert_and_trim_bb(img, r.rect) for r in cnn_rects] # coordinates of faces bb
    count = 0
    filename = Path(path).stem

    # in case there are multiple faces in the picture
    for x, y, w, h in cnn_boxes:
        img_new = img.copy()
        img_new = img_new[y : y + h, x : x + w]
        # Creates a new file
        with open(f"{save_path}{filename}_{count}.png", "w") as fp:
            pass
        cv2.imwrite(f"{save_path}{filename}_{count}.png", img_new)
        count += 1

# @TODO: apply version for multiple faces
def haarcascade(path):
    img, gray, rgb = load_img(path)
    haar_detector = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    start = time.perf_counter()
    rects = haar_detector.detectMultiScale(gray, scaleFactor=1.05,
	                                minNeighbors=5, minSize=(30, 30),
	                                flags=cv2.CASCADE_SCALE_IMAGE)
    img_haar = img.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(img_haar, (x, y), (x + w, y + h), (0, 255, 0), 2)
    finish = time.perf_counter()
    print("Time Haarcascade: " + str((finish - start)))
    return img_haar


if __name__ == "__main__":
    images_path = "../ExpW_test/images/"
    save_path = "faces/" # @TODO: zrob ladniej bo teraz trzeba dodawac samemu z lapy
    for file in os.listdir(images_path):
        print(file, images_path)
        cnn_crop(images_path + file, save_path)
