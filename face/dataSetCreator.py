import cv2
import os
from . import trainer


class DataSetCreator:
    def __init__(self, cam, identifier, captures, path):
        self.cam = cam
        self.identifier = identifier
        self.captures = captures
        self.path = str(path)  # full folder path must be specified
        self.capture()

    def capture(self):
        # haarcascade to be used
        faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        cam = cv2.VideoCapture(self.cam)  # cam inside the parenthesis is the index of the camera to be used

        terminator = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                terminator += 1
                cv2.imwrite(str(self.path) + '/' + str(self.identifier) + '.' + str(terminator) + '.jpg',
                            gray[y: y + h, x: x + w])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 150), 2)
                cv2.waitKey(100)


            cv2.imshow('dataSet Create and Train', img)
            if terminator >= self.captures:
                break


        # train here
        trainer.Trainer(self.path)

        cam.release()
        cv2.destroyAllWindows()
