# -*- coding:utf-8 -*-
# _*_ coding:utf-8 _*_

import numpy as np
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# # cv2Read image
img = cv2.imread("Trudeau.jpeg")


img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Face number rects
rects = detector(img_gray, 0)
for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
    for idx, point in enumerate(landmarks):
        # 68 points coordinates
        pos = (point[0, 0], point[0, 1])
        print(idx,pos)

        # Use cv2.circle to draw a circle for each feature point, a total of 68
        cv2.circle(img, pos, 5, color=(0, 255, 0))
        # Use cv2.putText output 1-68
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(idx+1), pos, font, 0.8, (0, 0, 255), 1,cv2.LINE_AA)

cv2.namedWindow("ShowPoints", 2)
cv2.imshow("img", img)

cv2.imwrite("Trudeau_points.jpg",img)

cv2.waitKey(0)