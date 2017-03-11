import cv2

from constants import INPUT_FILE, CLASSIFIER_PATH

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
