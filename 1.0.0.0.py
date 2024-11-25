import cv2 as cv
from math import sin,cos,pi
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

#1
cap = cv.VideoCapture("sparky.mp4")
 
while(1):
 
    # Take each frame
    _, frame = cap.read()
 
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
 
    # define range of blue color in HSV
    lower_blue = np.array([150,153,163])
    upper_blue = np.array([160,255,94])
 
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
 
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    
    plt.imshow(res)
    plt.show()
 
cap.release()
cv.destroyAllWindows()