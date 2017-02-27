import cv2
import numpy as np
from matplotlib import pyplot as plt

img_original = cv2.imread('guilhem3.jpg')
img_camera = cv2.imread('guilhem-pirate.jpg')

gray_original = img_original
gray_camera = img_camera

#gray_original = cv2.cvtColor(img_original,cv2.COLOR_BGR2GRAY)
#gray_camera = cv2.cvtColor(img_camera,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp_original, des_original = sift.detectAndCompute(gray_original,None)
kp_camera, des_camera = sift.detectAndCompute(gray_camera,None)

outImage = None

img = cv2.drawKeypoints(gray_original,kp_original, None)
img2 = cv2.drawKeypoints(gray_camera,kp_camera, None)


cv2.imwrite('sift_keypoints_original.jpg',img)
cv2.imwrite('sift_keypoints_camera.jpg',img2)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_original,des_camera, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.4*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img,kp_original,img2,kp_camera,good,None,flags=2)

plt.imshow(img3),plt.show()