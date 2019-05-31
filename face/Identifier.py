import cv2
import face_recognition
import os
from datetime import datetime


class Detect:
    def __init__(self, cam):
        self.cam = cam


    def identify(self):


        known_face_encodings = []

        known_face_names = []

        pathx = 'dataSetsample'

        dataSet_dir = os.listdir(pathx)

        if not os.path.exists('trans_history_captures'):
            os.mkdir('trans_history_captures')


        for i in dataSet_dir:
            try:
                image = face_recognition.load_image_file(pathx + '/' + i)
                face_encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(i.split('.')[0])
            except:
                pass

        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        video_capture = cv2.VideoCapture(self.cam)
        while True:
            ret, frame = video_capture.read()

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            rgb_small_frame = small_frame[:, :, ::-1]

            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                        cv2.imwrite('trans_history_captures/' + str(datetime.now()) + '.png', frame)
                        return name

                    face_names.append(name)
                    if name == "Unknown":
                        return False

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 00, 255), 2)

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            cv2.imshow('Face Recognizer and Identifier', frame)

            if cv2.waitKey(1) == 27:
                break

        video_capture.release()
        cv2.destroyAllWindows()



    # def check(self):
    #     faceDetect = cv2.CascadeClassifier('lbpcascade_profileface.xml')
    #     eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    #     cam = cv2.VideoCapture(self.cam)
    #
    #
    #     while True:
    #         ret, img = cam.read()
    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         face = faceDetect.detectMultiScale(gray, 1.2, 5)
    #
    #         for (x, y, w, h) in face:
    #             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #             cv2.putText(img, 'validating . . .' , (x, y + h),
    #                         cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv2.LINE_AA)
    #             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #             roi_gray = gray[y:y+h, x:x+w]
    #             eyes = eye_cascade.detectMultiScale(roi_gray)
    #             for (ex, ey, ew, eh) in eyes:
    #                 if ex and ey and ew and eh:
    #                     return True
    #         cv2.imshow('Face Recognizer and Identifier', img)
    #         if cv2.waitKey(1) == 27:
    #             break
    #     cam.release()
    #     cv2.destroyAllWindows()

