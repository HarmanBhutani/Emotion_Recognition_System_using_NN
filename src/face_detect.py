import cv2

from constants import INPUT_FILE, CLASSIFIER_PATH


def crop_face(image, rescale_factor=2):
    cascade = cv2.CascadeClassifier(CLASSIFIER_PATH)
    imageScaled = cv2.resize(image, (image.shape[0] / rescale_factor,
                                     image.shape[1] / rescale_factor))

    # The image might already be equalized, so no need for that here
    gray = cv2.equalizeHist(imageScaled)
    rects = cascade.detectMultiScale(gray, 1.1, 3)

    # We need to find exactly one face in the picture
    # print("len(rects)")
    print(len(rects))
    if len(rects) is not 1:
        return None

    x, y, w, h = map(lambda x: x * rescale_factor, rects[0])
    face = image[y:y + h, x:x + w]
    return face


capture = cv2.VideoCapture(INPUT_FILE)

classifier = cv2.CascadeClassifier(CLASSIFIER_PATH)

if capture.isOpened():
    ret, frames = capture.read()
    capture.set(cv2.CAP_PROP_POS_MSEC, 38000)
else:
    ret = False

while ret:
    ret, frames = capture.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 255, 0), 1)

    # Display frames in a window
    frames = cv2.resize(frames, (960, 540))
    cv2.imshow('video', frames)

    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()
