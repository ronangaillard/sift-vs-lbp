from scipy.cluster.vq import *
import numpy as np
import glob
import cv2
import scipy
from test_face_detect import PROCESSED_FACES_DIR

# Creating list of images 
print 'Getting images'
images = []
for infile in glob.glob(PROCESSED_FACES_DIR + '/*.jpg'):
    pic = cv2.imread(infile)
    images.append(pic)

np.random.shuffle(images)
my_set = images

print 'Number of images ', len(my_set)

# Split set
print 'Splitting set'
train = my_set[:len(my_set)/2]
test = my_set[len(my_set)/2:]

# Compute descriptors
print 'Computing descriptors'
sift = cv2.xfeatures2d.SIFT_create()

descriptors = np.array([])
for pic in train:
    kp, des = sift.detectAndCompute(pic, None)
    descriptors = np.append(descriptors, des)

desc = np.reshape(descriptors, (len(descriptors)/128, 128))
desc = np.float32(desc)

print desc

print "kmeans", scipy.cluster.vq.kmeans(desc, k_or_guess=1000, iter=20, thresh=1e-05)


