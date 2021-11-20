import os

from .face_alignment import image_align
from .landmarks_detector import LandmarksDetector


cur_dir = os.path.split(os.path.realpath(__file__))[0]
model_path = os.path.join(cur_dir, 'shape_predictor_68_face_landmarks.dat')


class FaceAlign:
    def __init__(self, predictor_model_path=model_path):
        self.landmarks_detector = LandmarksDetector(predictor_model_path)

    def get_crop_image(self, image):
        lms = []
        for i, face_landmarks in enumerate(self.landmarks_detector.get_landmarks(image), start=1):
            lms.append(face_landmarks)
        if len(lms) < 1:
            return None
        out_image = image_align(image, lms[0])

        return out_image

