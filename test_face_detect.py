import numpy as np
import cv2 
from matplotlib import pyplot as plt
import os

OPENCV_PATH = '/usr/local/Cellar/opencv3/HEAD-dcbed8d/share/OpenCV/haarcascades/'

face_cascade = cv2.CascadeClassifier(OPENCV_PATH + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(OPENCV_PATH + 'haarcascade_eye.xml')

i = 0
for root, dirs, files in os.walk('./photos/'):
    data = [os.path.join(root,f) for f in files if f.endswith(".jpg")]
    for fi in data:
        img = cv2.imread(fi)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            crop_img = img[y:y+h, x:x+w]
            cv2.imwrite('./faces_processed/' + str(i) +'.jpg', crop_img)
            i+=1
