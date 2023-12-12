import cv2
import dlib
import os
from pathlib import Path


def load_img(path):
    """
    loads image and changes channel colour
    """
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, rgb


def convert_and_trim_bb(image, rect):
    """
    reads bounding box corner coordinates
    and converts it to coordinates of left upper corner
    and width and height
    """
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return (startX, startY, w, h)


def cnn_crop(path, save_path):
    """
    detects face from image using cnn algorithm
    """
    img, rgb = load_img(path)
    cnn_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
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


if __name__ == "__main__":
    images_path = "../ExpW_test/images/"
    save_path = "faces/" # @TODO: zrob ladniej bo teraz trzeba dodawac samemu z lapy
    for file in os.listdir(images_path):
        print(file, images_path)
        cnn_crop(images_path + file, save_path)
