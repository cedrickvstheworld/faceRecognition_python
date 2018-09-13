import cv2


class Detect:
    def __init__(self, cam):
        self.cam = cam
        self.identify()

    def identify(self):
        faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        cam = cv2.VideoCapture(self.cam)
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
            face_recognizer.read('processedData/facesData.yml')
        except:
            print('no dataSet to be read')
            exit()

        # confidence = 0
        # ID = None

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = faceDetect.detectMultiScale(gray, 1.2, 5)
            ID = None

            for (x, y, w, h) in face:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 150), 2)
                ID, confidence = face_recognizer.predict(gray[x: x + w, y: y + h])
                conf = str(confidence).split('.')[0]

                # 0 is the perfect match, 50 is better than 100
                # NOTE: the better the webcam's quality, the lesser the negatives
                if confidence <= 35:
                    cv2.putText(img, 'ID:' + str(ID) + '  ' + 'negatives:' + conf, (x, y + h),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv2.LINE_AA)
                    return ID
                else:
                    cv2.putText(img, 'recognizing ...  ' + conf, (x, y + h),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)


            cv2.imshow('Face Recognizer and Identifier', img)
            if cv2.waitKey(100) == 27:
                quit()
                exit()
                break

            # or
            #
            # if confidence <= 80:
            #     return True, ID

        cam.release()
        cv2.destroyAllWindows()
