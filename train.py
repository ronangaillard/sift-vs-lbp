from scipy.cluster.vq import *
import numpy as np
import glob
import cv2
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
descriptors = [sift.detectAndCompute(pic, None) for pic in train]


