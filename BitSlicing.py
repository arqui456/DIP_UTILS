# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:23:25 2018

@author: dudummv
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def doNothing(meh):
	pass

cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO)

img = cv2.imread("img/dollar.tif", cv2.IMREAD_GRAYSCALE)
if img is None:
    print('Could not open or find the image')
    exit(0)
img2 = np.copy(img)

slice = 1
cv2.createTrackbar("slice", "img2", slice, 7, doNothing)

while True:
    slice = cv2.getTrackbarPos("slice", "img2")
    img2 = cv2.bitwise_and(img, 1 << slice, img2)

    img2 = np.asarray(img2, np.float32)
    cv2.normalize(img2, img2, 0, 1, cv2.NORM_MINMAX)
    cv2.imshow("img", img)
    cv2.imshow("img2", img2)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()