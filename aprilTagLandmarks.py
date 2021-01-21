from pupil_apriltags import Detector
import numpy as np
import cv2

tagLibrary = "tag36h11"

class apriltagDetector():
    def __init__(self,colorIntrinsics,tagSize=.161,tagLibrary='tag36h11'):
        self.colorInt = colorIntrinsics
        self.tagLibrary = tagLibrary
        self.tagSize = tagSize
        self.detector = Detector(families=self.tagLibrary,
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

    def detect(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(img,estimate_tag_pose=True,
                camera_params=self.colorInt, tag_size=self.tagSize)
