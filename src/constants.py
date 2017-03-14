# Constants used in the Projects

INPUT_FILE = "input.MKV"
CLASSIFIER_PATH = "classifiers/haarcascade_frontalface_default.xml"

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
SIZE_FACE = 48

DATA_SET_DIR = "dataset"
DATA_SET_FILE = DATA_SET_DIR + "/fer2013.csv"
TRAINING_SET = DATA_SET_DIR + "/data_set_fer2013.npy"
TRAINING_LABELS = DATA_SET_DIR + "/data_labels_fer2013.npy"

TEST_SET = DATA_SET_DIR + "/test_set_fer2013.npy"
TEST_LABELS = DATA_SET_DIR + "/test_labels_fer2013.npy"
