import cv2
import numpy as np
import serial
import time
import sys
from serial import SerialException
import struct
import math
import matplotlib.pylab as plot
import argparse
from collections import deque


cap = cv2.VideoCapture(0)


# SET FRAME SIZE AND FRAME RATE
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)	
cap.set(cv2.CAP_PROP_FPS, 60)





# IMAGE DISTANCE CALIBRATION DATA
KNOWN_DISTANCE = 30
KNOWN_WIDTH = 20
IMAGE_PATHS = ["/home/elliott/turret/30cm_image_3.png", "/home/elliott/turret/60cm_image_3.png", "/home/elliott/turret/90cm_image_3.png", "/home/elliott/turret/150cm_image_3.png"] 	# Calibration image may be slightly off



# FUNCTION FOR INTIAL CALIBRATION TO FIND THE TARGET DISTANCE
def find_marker(image):
	

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(hsv, (42,34,71),(75,255,158))	#mask the image with the hsv values for the square

        # Additional thresholding, erode & dilate, removes noise
	mask = cv2.erode(mask, None, iterations=0)
	mask = cv2.dilate(mask, None, iterations=0)

        # Bitwise-AND mask and original image
	res = cv2.bitwise_and(image,image, mask= mask)


        # Find contours (A ball)
	contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]

	c = max(contours, key=cv2.contourArea)
	rect = cv2.minAreaRect(c)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(image, [box], 0, (0, 255, 255), 2)

 
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)


# FUNCTION TO CALCULATE DISTANCE OF TARGET FROM CAMERA
def distance_to_camera(knownWidth, focalLength, perWidth):

	if perWidth != 0:
		# compute and return the distance from the maker to the camera
		return (knownWidth * focalLength) / perWidth
	else:
		return 0








	

# CALIBRATE CAMERA
image = cv2.imread(IMAGE_PATHS[0])
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

#centimetres = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

#box = cv2.boxPoints(marker)


for imagePath in IMAGE_PATHS:
	
	image = cv2.imread(imagePath)
	marker = find_marker(image)
	centimetres = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
	
	box = np.int0(cv2.boxPoints(marker))
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.putText(image, "Distance to camera: %.2fcm" % centimetres,(5,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,.5,(225,0,0))
	
	cv2.imshow("image", image)
	cv2.waitKey(0)




			
cv2.destroyAllWindows()
