import numpy as np
import cv2 
from matplotlib import pyplot as plt
import os
import subprocess as sp
import re
import time

FFMPEG_BIN = 'ffmpeg'
video_in = './video/MOV_0673.mp4'
OPENCV_PATH = '/usr/local/Cellar/opencv3/HEAD-dcbed8d/share/OpenCV/haarcascades/'
PROCESSED_FACES_DIR = './faces_processed'

face_cascade = cv2.CascadeClassifier(OPENCV_PATH + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(OPENCV_PATH + 'haarcascade_eye.xml')

ffprobe_command = ['ffprobe', 
    '-v', 'error', 
    '-of', 'flat=s=_', 
    '-select_streams', 'v:0', 
    '-show_entries', 'stream=width,height', 
    video_in]
ffprobe_pipe = sp.Popen(ffprobe_command, stdout = sp.PIPE, bufsize=10**8)
re_match = re.findall('(?<==)\d+', ffprobe_pipe.stdout.read())
height = int(re_match[0])
width = int(re_match[1])
ffprobe_pipe.stdout.flush()


mpeg_command = [ FFMPEG_BIN,
            '-i', video_in,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
pipe = sp.Popen(mpeg_command, stdout = sp.PIPE, bufsize=10**8)


image_index = 0
while pipe.stdout:
    raw_image = pipe.stdout.read(height*width*3)
    # transform the byte read into a numpy array
    img =  np.fromstring(raw_image, dtype='uint8')
    if len(img) < height*width*3:
        pipe.terminate()
        break
    img = img.reshape((width,height,3))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        crop_img = img[y:y+h, x:x+w]
        cv2.imwrite(PROCESSED_FACES_DIR + '/' + str(image_index) +'.jpg', crop_img)
        image_index += 1
    # throw away the data in the pipe's buffer.
exit()



if not os.path.exists(PROCESSED_FACES_DIR):
    os.makedirs(PROCESSED_FACES_DIR)

i = 0
for root, dirs, files in os.walk('./photos/'):
    data = [os.path.join(root,f) for f in files if f.endswith(".jpg")]
    for fi in data:
        img = cv2.imread(fi)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            crop_img = img[y:y+h, x:x+w]
            cv2.imwrite(PROCESSED_FACES_DIR + '/' + str(i) +'.jpg', crop_img)
            i+=1
