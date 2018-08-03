import cv2


class Detect:
    def __init__(self, cam):
        self.cam = cam
        self.detector()

    def detector(self):
        faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cam = cv2.VideoCapture(self.cam)
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
         face_recognizer.read('processedData\\facesData.yml')
        except:
            print('no dataSet to be read')
            exit()

        # confidence = 0
        # ID = None

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = faceDetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in face:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 150), 2)
                ID, confidence = face_recognizer.predict(gray[x: x + w, y: y + h])

                if confidence >= 50 and ID is not None:
                    conf = str(confidence).split('.')[0]
                    cv2.putText(img, 'ID:' + str(ID) + ' ' + conf + '%', (x, y + h),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(img, 'unrecognized', (x, y + h),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('Face Recognizer and Identifier', img)
            if cv2.waitKey(1) == ord('q'):
                break

            # or
            #
            # if confidence >= 70:
            #     return True, ID
            
        cam.release()
        cv2.destroyAllWindows()
