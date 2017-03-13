import pandas as pd

from constants import *
import cv2

classifier = cv2.CascadeClassifier(CLASSIFIER_PATH)

dataSet = pd.read_csv(DATA_SET_PATH)

training_labels = []
training_image = []
test_labels = []
test_image = []
index = 1
total = dataSet.shape[0]


def cropFace(image, rescaleForReconigtion=2):
    cascade = cv2.CascadeClassifier(CLASSIFIER_PATH)
    imageScaled = cv2.resize(image, (image.shape[0] / rescaleForReconigtion,
                                     image.shape[1] / rescaleForReconigtion))

    # The image might already be equalized, so no need for that here
    gray = cv2.equalizeHist(imageScaled)
    rects = cascade.detectMultiScale(gray, 1.1, 3)

    # You need to find exactly one face in the picture
    # print("len(rects)")
    print(len(rects))
    if len(rects) is not 1:
        return None

    x, y, w, h = map(lambda x: x * rescaleForReconigtion, rects[0])
    face = image[y:y + h, x:x + w]
    return face
