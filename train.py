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
    pic_gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(pic_gray, None)
    descriptors = np.append(descriptors, des)

desc = np.reshape(descriptors, (len(descriptors)/128, 128))
desc = np.float32(desc)

print desc

kmeans = scipy.cluster.vq.kmeans(desc, k_or_guess=1000, iter=20, thresh=1e-05)[0]

# Test
for image in test:
    bf = cv2.BFMatcher()

    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    kp_image, des_image = sift.detectAndCompute(gray_image,None)
    matches = bf.knnMatch(kmeans, des_image, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.4*n.distance:
            good.append([m])

    print 'Number of good points : ', len(good)

   


