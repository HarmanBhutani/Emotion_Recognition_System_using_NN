import tflearn
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d

import csv_to_numpy
from constants import *
from dataset_loader import DatasetLoader


class EmotionRecognition:
    def __init__(self):
        self.dataset = DatasetLoader()

    def build_network(self):
        print('[+] Building CNN')
        self.network = input_data(shape=[None, SIZE_FACE, SIZE_FACE, 1])
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 128, 4, activation='relu')
        self.network = dropout(self.network, 0.3)
        self.network = fully_connected(self.network, 3072, activation='relu')
        self.network = fully_connected(self.network, len(EMOTIONS), activation='softmax')
        self.network = regression(self.network, optimizer='momentum', loss='categorical_crossentropy')

        self.model = tflearn.DNN(self.network, checkpoint_path=DATA_SET_DIR + '/emotion_recognition', max_checkpoints=1,
                                 tensorboard_verbose=2)
