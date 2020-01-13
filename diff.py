import numpy as np
import cv2
import imutils
from skimage.metrics import structural_similarity
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-f', '--first', required = True, help = 'image one')
ap.add_argument('-s', '--second', required = True, help = 'image two')

args = vars(ap.parse_args())



### Load in the images and convert to gray
def load(image1, image2, gray = False):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    img1_ = cv2.resize(img1, (300,300))
    img2_ = cv2.resize(img2, (300, 300))
    if gray == True:
        gray1 = cv2.cvtColor(img1_, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_, cv2.COLOR_BGR2GRAY)
    return img1_, img2_, gray1, gray2

def compare(gray1, gray2):
    (score, difference) = structural_similarity(gray1, gray2, full = True)
    difference = (difference*255).astype('uint8')
    return score, difference

def find_contours(difference, image1, image2):
    threshold = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for cnt in cnts:

        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(image1, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.rectangle(image2, (x, y), (x+w, y+h), (0,0,255), 2)

    cv2.imshow('Original Image', image1)
    cv2.imshow('Altered Image', image2)
    cv2.imshow('difference btween the two images', difference)
    cv2.imshow('threshold', threshold)
    cv2.waitKey(0)

imageA, imageB, grayA, grayB = load(args["first"], args["second"], gray = True)
score, diff = compare(grayA, grayB)
find_contours(diff, imageA, imageB)
