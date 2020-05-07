'''
    copyright Peter Milani 2020

    Released under GPL 3
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    '''

import cv2
import numpy as np
import os
import argparse
from scipy.spatial.distance import cdist

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.75
def threshold(im, sdthres=30):
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    imBackground = np.ones(imGray.shape)
    imBackground[sdthres < (imGray- np.median(imGray))/imGray.std()] = 0
    imStars = np.zeros(imGray.shape)
    #print('stdev {} mean {} median {}'.format(imGray.std(), imGray.mean(), np.median(imGray)))
    imStars[sdthres < (imGray- np.median(imGray))/imGray.std()] = 25
    return imBackground, imStars

def findBlobs(imstars):

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold=1
    params.maxThreshold=20
    params.thresholdStep=2
    params.minArea=20

    params.filterByArea=True
    params.minDistBetweenBlobs=10
    params.filterByInertia=False
    params.filterByConvexity=False
    params.filterByCircularity=False
    params.filterByColor=False
    
    
    detector = cv2.SimpleBlobDetector_create(params)
    #print(imstars)
    keypoints = detector.detect(imstars.astype(np.uint8))
    k = keypointAsVector(keypoints)
    
    des = cdist(k, k, 'euclidean')
    np.matrix.sort(des, axis=1)
    
    return keypoints, des


def keypointAsVector(keypoints):

    kparray = np.zeros((len(keypoints), 2))
    for i,  k in enumerate(keypoints):
        kparray[i, 0]=k.pt[0]
        kparray[i, 1]=k.pt[1]

    kparray.astype(np.int32)
    return kparray

def findStars(im1, im2):
    
    im1bg, im1stars = threshold(im1,3)
    im2bg, im2stars = threshold(im2, 3)
    cv2.imshow('img1', im1stars)
    cv2.imshow('img2', im2stars)
    #cv2.waitKey(0)
    #kp1 = findBlobs(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY))
    kp1, des1 = findBlobs(im1stars)
    kp2, des2 = findBlobs(im2stars)
    

    

    im_with_keypoints = cv2.drawKeypoints(im1, kp1, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("img3", im_with_keypoints)
    #cv2.waitKey(0)
    #TODO Create distance descriptors
    return kp1, des1, kp2, des2

def getOrbKP(im1, im2):

    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, des1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, des2 = orb.detectAndCompute(im2Gray, None)

    return keypoints1, des1, keypoints2, des2

def appendOrbDescWithPos(des, kps):
    kparray = np.zeros((des.shape[0], 2))
    for i,  k in enumerate(kps):
        kparray[i, 0]=k.pt[0]
        kparray[i, 1]=k.pt[1]

    kparray.astype(np.int32)
    
    d = np.hstack((kparray, des ))

    #print(d.astype(np.int32))
    return kparray 

def getMatchedFeatures(im1, im2):
    th=30
    # Convert images to grayscale
    keypoints1, d1, keypoints2, d2= findStars(im1, im2)
    
    #print(d1) 
    # Match features.
    matcher = cv2.BFMatcher()
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = matcher.knnMatch(d1[:, :6].astype(np.float32), d2[:,:6].astype(np.float32), k=2)

    #matches = matcher.match(d1[:,:6].astype(np.float32), d2[:, :6].astype(np.float32), None)
    # Sort matches by score
    matches.sort(key=lambda x: x[0].distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    good = []
    good_without_list = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
        good_without_list.append(m)
    matches=good_without_list
    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    cv2.imshow('img2', imMatches)
    cv2.waitKey(1)
    return keypoints1, keypoints2, matches, imMatches


def alignImages(im1, im2):
    keypoints1, keypoints2, matches, imMatches = getMatchedFeatures(im1, im2)
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
        
    #print("{}, {}".format(points1[:5,:], points2[:5,:]))

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    print(h)

    if h is not None:
        if h[:,2].mean() > 1000:
            return None, None
        #cv2.imwrite("matches.jpg", imMatches)
        # Use homography
        height, width, channels = im2.shape
        
        im1Reg = cv2.warpPerspective(im1, h, (width, height))
        cv2.imshow('img1',im1)
        cv2.imshow('img2',im1Reg)
        #cv2.waitKey(0)
        return im1Reg, h
    else:
        return None, None

def findBadPixels(im):
    '''
    These seem to be when there is an unbalanced value
    across the colour channels
    '''

def parse_args():
    parser=argparse.ArgumentParser(description='Stack a series of images captured by a camera of the night sky')
    parser.add_argument('-d', '--dir', dest='dir', help='directory to the images from a run')
    return parser.parse_args()


if __name__ == "__main__":

    cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
    
    args = parse_args()

    dir = os.path.abspath(args.dir)
    files = os.listdir(dir)
    print(files)
    pictures=[]
    for f in files:
        if '.jpg' in f:
            pictures.append(f)

    imgref = cv2.imread('{}/{}'.format(dir, pictures[int(len(pictures)/2)]))
    #imgref = cv2.resize(imgref,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)

    images =[]
    unaligned=[]
    for p in pictures:
        img=cv2.imread('{}/{}'.format(dir, p))

        imgaligned, h = alignImages(img, imgref)
        if imgaligned is not None:
           images.append(imgaligned)
           print('dtype img {}'.format(imgaligned.dtype))
        else:
           unaligned.append(img)
           #imgref=img
        #cv2.waitKey(0)

    com=imgref.astype(np.float32)
    img1 = imgref.astype(np.float32)

    for img in images:
        img = cv2.medianBlur(img, 7)
        img1+=img.astype(np.float32)

        if img1.max() > 220.:
            img1*=220.0/img1.max()

    
    img1*=255.0/img1.max()
    cv2.imshow('img1', img1.astype(np.uint8))
    cv2.waitKey(0)




