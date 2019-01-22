import cv2
import numpy as np
import serial
import time
import sys
from serial import SerialException
import struct
import math

cap = cv2.VideoCapture(0)

try:
	ser = serial.Serial('/dev/ttyACM0')
except SerialException:
	print('Port already open or access is denied')
	sys.exit()

ser.write("")

print(ser.name)

# SET FRAME SIZE AND FRAME RATE
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)	
cap.set(cv2.CAP_PROP_FPS, 180)


# OBJECT COLOR RANGES
color_ranges = [
    ((28,49,77),(72,255,204), 'square'),
    ((0,78,177),(88,255,255), 'ball')]


# IMAGE DISTANCE CALIBRATION DATA
KNOWN_DISTANCE = 30
KNOWN_WIDTH = 7
IMAGE_PATH = ["/home/student/turret/30cm_image.png"]

# INITIAL FRAME COUNTER
frameCounter = 0



def draw_circle(contours, center):
    # If contours are found
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        # This line calculates centroid
	if M["m00"] != 0:
        	draw_circle.center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 1:
            #cv2.circle(img, center, radius, color, thickness=1, lineType=8, shift=0)
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, draw_circle.center, 5, (0, 0, 255), -1)
            print('circle', draw_circle.center)


def draw_rect(frameCounter):
    # If contours are found
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
	box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)
	
        # box prints out array as the following:
	#[ [bottom left  (x,y)]
	#  [top left     (x,y)]
	#  [top right    (x,y)]
	#  [bottom right (x,y)] ]

	# box[X] can be assigned to their corresponding positions
	draw_rect.bl_pos = box[0]
	draw_rect.tl_pos = box[1]
	draw_rect.tr_pos = box[2]
	draw_rect.br_pos = box[3]

	# Each position is an array with X and Y coordinates
	# Now calculate the center X coordinate
	center_top_x_pos = (draw_rect.tl_pos[0] + draw_rect.tr_pos[0]) / 2
	center_bottom_x_pos = (draw_rect.bl_pos[0] + draw_rect.br_pos[0]) / 2
	center_x_pos = (center_top_x_pos + center_bottom_x_pos) / 2
	#print(center_x_pos)
	
	# Now calculate the center Y coordinate
	center_left_y_pos = (draw_rect.tl_pos[1] + draw_rect.bl_pos[1]) / 2
	center_right_y_pos = (draw_rect.tr_pos[1] + draw_rect.br_pos[1]) / 2
	center_y_pos = (center_left_y_pos + center_right_y_pos) / 2
	print('square', center_x_pos, center_y_pos)
	
	# Draw center point
	cv2.circle(frame, (center_x_pos, center_y_pos), 8, (255, 0, 0), -1)

	# Calculate distance of rectangle to camera. rect[1][0] is the width of the rectangle in pixals
	dis_to_cam_cm = distance_to_camera(KNOWN_WIDTH, focalLength, rect[1][0])

	cv2.putText(frame, "Distance to camera: %.2fcm" % dis_to_cam_cm,(5,10),cv2.FONT_HERSHEY_COMPLEX_SMALL,.5,(225,0,0))
	
	# Calculate distance of rectangle to middle of the frame. rect[1][0] is the width of the rectangle in pixals
	dis_from_center_cm = distance_from_center(center_x_pos, KNOWN_WIDTH, rect[1][0])

	cv2.putText(frame, "Distance from center: %.2fcm" % dis_to_center_cm,(5,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,.5,(225,0,0))
	
	# Calculate the angle the target is from the center
	xServoAngle = math.asin(dis_from_center_cm / dis_to_cam_cm)
	xServoAngle = math.degrees(xServoAngle)
	print(xServoAngle)

	# Write the angle to the xServo every 60 frames
	if frameCounter == 60:	

		center_x_pos_as_string = str(xServoAngle)
		ser.write(b'xServo')
		print('xServo')
		time.sleep(0.03)
		ser.write(center_x_pos_as_string)
		print(center_x_pos_as_string)
		time.sleep(0.03)


def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
 
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
	c = max(contours, key = cv2.contourArea)
 
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)


def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth


def distance_from_center(center_x_pos, knownWidth, perWidth):

	if center_x_pos > 320:
		distance_from_center = center_x_pos - 320
	else:
		distance_from_center = -(320 - center_x_pos)

	cm_per_pixal = knownWidth / perWidth
	
	return distance_from_center * cm_per_pixal
	

def hit_box():
    # if ball_x is in range of square x1, x2?
    if draw_rect.tl_pos[0] < draw_circle.center[0] < draw_rect.tr_pos[0] and draw_rect.tl_pos[1] < draw_circle.center[1] < draw_rect.bl_pos[1]:
    # and if ball_Y is in range of square y
        print('HIT!')
    # Doesn't account for square rotation...



# CALIBRATE CAMERA
image = cv2.imread(IMAGE_PATH)
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

centimetres = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

box = cv2.boxPoints(marker)
box = np.int0(box)
cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

cv2.putText(frame, "Distance from center: %.2fcm" % dis_to_center_cm,(5,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,.5,(225,0,0))

cv2.imshow("image", image)
cv2.waitKey(0)



	

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   

# FIND THE BALL AND MASK IT

    # define range of color in HSV
    lower_bounds = np.array([120,117,91])
    upper_bounds = np.array([179,255,255])

    #find_ball(hsv, lower_bounds, upper_bounds, mask)

    # Threshold the HSV image to get only specific colors
    mask = cv2.inRange(hsv, lower_bounds, upper_bounds)

    # Additional thresholding, erode & dilate, removes noise
    mask = cv2.erode(mask, None, iterations=7)
    mask = cv2.dilate(mask, None, iterations=7)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)



# DRAW A CIRCLE AROUND THE BALL

    # Find contours (A ball)
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    draw_circle(contours, center)

    frameCounter = frameCounter + 1

    draw_rect(frameCounter)

    if frameCounter == 60:
	frameCounter = 0
    



# SHOW THE FRAME

    #cv2.imshow('frame',frame)
    # These masks block out all but the ball, including the circles.
    # solution is to declare circle vars out of if statement
    cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    cv2.imshow('yay', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
