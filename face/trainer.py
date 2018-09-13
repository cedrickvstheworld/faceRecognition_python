import cv2
import numpy as np
import os
from PIL import Image


class Trainer:
    def __init__(self, path):
        face_recognizer = cv2.face_LBPHFaceRecognizer.create()
        self.path = str(path)

        trained_data_dir = 'processedData'

        if not os.path.exists(trained_data_dir):
            os.mkdir(trained_data_dir)

        identifier, faces = self.write_images()
        face_recognizer.train(faces, identifier)
        face_recognizer.save(trained_data_dir + '/facesData.yml')

    def write_images(self):
        imagePaths = [os.path.join(self.path, f) for f in os.listdir(self.path)]
        faces = []
        identifier = []

        for i in imagePaths:
            read_face = Image.open(i).convert('L')
            face_array = np.array(read_face, dtype='uint8')
            idx = int(os.path.split(i)[-1].split('.')[0])
            faces.append(face_array)
            identifier.append(idx)
            cv2.waitKey(10)

        return np.array(identifier), faces
