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
