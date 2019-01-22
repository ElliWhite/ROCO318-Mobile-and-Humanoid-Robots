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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())


cap = cv2.VideoCapture(0)


# SET FRAME SIZE AND FRAME RATE
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)	
cap.set(cv2.CAP_PROP_FPS, 60)


# OBJECT COLOR RANGES
color_ranges = [
    ((33,43,115),(98,255,175), 'square'),
    ((96,148,136),(130,255,255), 'ball')]


# IMAGE DISTANCE CALIBRATION DATA
KNOWN_DISTANCE = 30
KNOWN_WIDTH = 20
IMAGE_PATH = "/home/elliott/turret/30cm_image_2.png"	# Calibration image may be slightly off

BALL_TO_FLOOR_DISTANCE = 22.5 #CM
#CAMERA_TO_FLOOR_DISTANCE = 30 #CM 
CAMERA_TO_BALL_DISTANCE = 40 #CM

# INITIALISE FRAME COUNTER
frameCounter = 0

# INITIALISE GRAPH
graph1 = plot.figure()
v = 5			# initial velocity of the ball. Declared as we don't know it

# INITIALISE VALUES FOR HIT OR MISS TIMERS
startTime = 0
timeToLaunch = 0.32		#Value in seconds for ball to be launched out of turret

pts = deque(maxlen=args["buffer"])


# FUNCTION FOR INTIAL CALIBRATION TO FIND THE TARGET DISTANCE
def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#gray = cv2.GaussianBlur(gray, (1, 1), 0)
	#edged = cv2.Canny(gray, 0, 100)
 
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	#contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
       #cv2.CHAIN_APPROX_SIMPLE)[-2]
	#c = max(contours, key = cv2.contourArea)

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(hsv, (38,28,65),(84,162,255))

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


# FUNCTION TO CALCULATE DISTANCE OF TARGET FROM CENTER OF FRAMCE
def distance_from_center(center_x_pos, knownWidth, perWidth):

	if perWidth != 0:
		if center_x_pos > 320:
			distance_from_center = center_x_pos - 320
		else:
			distance_from_center = -(320 - center_x_pos)

		if perWidth != 0:
			cm_per_pixal = knownWidth / perWidth
		else:
			cm_per_pixal = 0
	
		return distance_from_center * cm_per_pixal
	else:
		return 0


# FUNCTION TO CALCULATE DISTANCE OF TARGET TO TURRET
def distance_to_turret_calc(dis_to_cam_cm):

	#a**2 + b**2 = c**2

	dis_to_turr_base_sq = (dis_to_cam_cm ** 2) - ((BALL_TO_FLOOR_DISTANCE + CAMERA_TO_BALL_DISTANCE) ** 2)
	
	if (dis_to_turr_base_sq > 0):
		dis_to_turr_sq = dis_to_turr_base_sq + (BALL_TO_FLOOR_DISTANCE ** 2)
		return math.sqrt(dis_to_turr_sq)
	else:
		return 0

# FUNCTION TO CALCULATE DISTANCE OF TARGET TO TURRET BASE
def distance_to_turret_base_calc(dis_to_turret):

	dis_to_turr_base_sq = (dis_to_turret ** 2) - (BALL_TO_FLOOR_DISTANCE ** 2)
	
	if dis_to_turr_base_sq > 0:

		return math.sqrt(dis_to_turr_base_sq)
		
	else: 
		return 0


# FUNCTION TO CALCULATE THE ySERVO ANGLE
def calculate_yservo_angle(dis_to_turret_base_m):
	
	#Calulated using equation on Wikipedia
	BALL_TO_FLOOR_DISTANCE_M = BALL_TO_FLOOR_DISTANCE / 100

	tmp1 = (9.81 * (dis_to_turret_base_m ** 2)) + (2 * BALL_TO_FLOOR_DISTANCE_M * (v ** 2))
	
	if (9.81 * tmp1) < (v ** 4):
		tmp2 = math.sqrt((v ** 4) - (9.81 * tmp1))
		tmp3 = ((v ** 2) + tmp2) / (9.81 * dis_to_turret_base_m)
		theta = math.atan(tmp3)

		return theta
	else:
		return 0
		

# FUNCTION TO CALCULATE THE TIME OF FLIGHT
def calculate_time_of_flight(theta, BALL_TO_FLOOR_DISTANCE):
	
	#Calculated using equation on Wikipedia
	return (2 * v * math.sin(theta)) / 9.81

	

# FUNCTION TO DRAW TRAJECTORY GRAPH
def draw_graph(theta):

	#theta = math.pi/2 - theta

	t = np.linspace(0, 2, num=500) # Set time as 'continous' parameter.
	
	#for i in theta: # Calculate trajectory for every angle
	x1 = []
	y1 = []
	for k in t:
		x = ((v*k)*math.cos(theta)) 
		y = ((v*k)*math.sin(theta))-((0.5*9.81)*(k*k))
		x1.append(x)
		y1.append(y)
	p = [i for i, j in enumerate(y1) if j < 0] 

	for i in sorted(p, reverse = True):
		del x1[i]
		del y1[i]

	plot.axis([0.0,2.5, 0.0,2.0])
	ax = plot.gca()
	ax.set_autoscale_on(False)

	theta_deg = math.degrees(theta)	
	plot.plot(x1, y1, label='yServoAngle = %f' % theta_deg) # Plot for every angle
	plot.legend(loc='upper right')
	plot.ion()
	plot.pause(0.000000001)
	plot.draw() # And show on one graphic
	graph1.clear()


# FUNCTION TO DRAW CIRCLE AROUND BALL
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
			
			# update the points queue
			pts.appendleft(draw_circle.center)

			if radius > 1:
				#cv2.circle(img, center, radius, color, thickness=1, lineType=8, shift=0)
				cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
				cv2.circle(frame, draw_circle.center, 5, (0, 0, 255), -1)
				#print('circle', draw_circle.center)
				
			# loop over the set of tracked points
			for i in xrange(1, len(pts)):
				# if either of the tracked points are None, ignore
				# them
				if pts[i - 1] is None or pts[i] is None:
					continue
 
				# otherwise, compute the thickness of the line and
				# draw the connecting lines
				#thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
				thickness = 5
				cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)


# FUNCTION TO DRAW RECTANGLE AROUND TARGET AND CALCULATE ANGLE TO SEND TO xSERVO
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
	#print('square', center_x_pos, center_y_pos)
	
	# Draw center point
	cv2.circle(frame, (center_x_pos, center_y_pos), 8, (255, 0, 0), -1)

	
	
	# Calculate distance of rectangle to middle of the frame. rect[1][0] is the width of the rectangle in pixals
	dis_from_center_cm = distance_from_center(center_x_pos, KNOWN_WIDTH, rect[1][0])
	
	# Calculate distance of rectangle to camera. rect[1][0] is the width of the rectangle in pixals
	dis_to_cam_cm = distance_to_camera(KNOWN_WIDTH, focalLength, rect[1][0])
	
	dis_to_turr_cm = distance_to_turret_calc(dis_to_cam_cm)

	dis_to_turr_base_cm = distance_to_turret_base_calc(dis_to_turr_cm)
	
	dis_to_turr_base_m = dis_to_turr_base_cm / 100


	cv2.putText(frame, "Distance to camera: %.2fcm" % dis_to_cam_cm,(5,10),cv2.FONT_HERSHEY_COMPLEX_SMALL,.5,(225,0,0))
	cv2.putText(frame, "Distance from center: %.2fcm" % dis_from_center_cm,(5,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,.5,(225,0,0))
	cv2.putText(frame, "Distance to turret base: %.2fcm" % dis_to_turr_base_cm,(5,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,.5,(225,0,0))
	
	# Calculate the angle needed for the xServo
	if ((dis_from_center_cm / dis_to_turr_cm) < 1) and ((dis_from_center_cm / dis_to_turr_cm) > -1):
		if dis_to_turr_cm != 0:
			xServoAngle = math.asin(dis_from_center_cm / dis_to_turr_cm)
			xServoAngle = math.degrees(xServoAngle)
	else:
		xServoAngle = 0
	
	# Calculate the angle needed for the yServo
	if dis_to_turr_base_cm > 0:
		
		theta = calculate_yservo_angle(dis_to_turr_base_m)	# Need to change to take into account the height of turret before this
		theta = math.degrees(theta)
		
		if theta != 0:
			yServoAngle = theta
			
			draw_rect.time_of_flight = calculate_time_of_flight(theta, BALL_TO_FLOOR_DISTANCE)
			
			cv2.putText(frame, "Time of flight: %.2fS" % draw_rect.time_of_flight,(5,55),cv2.FONT_HERSHEY_COMPLEX_SMALL,.5,(225,0,0))
		
			if frameCounter == 120:
				theta = math.radians(theta)
				draw_graph(theta)
		else:
			print("ERROR - TARGET TOO FAR")
			yServoAngle = 0

	else:
		print("ERROR - TARGET TOO CLOSE")
		yServoAngle = 0
	
	
			
	# Write the angles to the Servos every 60 frames
	if frameCounter == 60:	
		xServoAngle_as_string = str(xServoAngle)
		ser.write(b'xServo')
		print('xServo')
		time.sleep(0.03)
		ser.write(xServoAngle_as_string)
		print(xServoAngle_as_string)
		time.sleep(0.03)
	
	if frameCounter == 120:
		center_y_pos_as_string = str(yServoAngle)
		ser.write(b'yServo')
		print('yServo')
		time.sleep(0.03)
		ser.write(center_y_pos_as_string)
		print(center_y_pos_as_string)
		time.sleep(0.03)

	
# FUNCTION TO DETECT IF BALL HAS HIT TARGET
def hit_box():
    # if ball_x is in range of square x1, x2?
    print(draw_circle.center)
    if draw_circle.center != 0:
		if draw_rect.tl_pos[0] < draw_circle.center[0] < draw_rect.tr_pos[0] and draw_rect.tl_pos[1] < draw_circle.center[1] < draw_rect.bl_pos[1]:
		# and if ball_Y is in range of square y
			print('HIT!')
		# Doesn't account for square rotation...
		else:
			print('MISS!')



# CALIBRATE CAMERA
image = cv2.imread(IMAGE_PATH)
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

centimetres = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

box = cv2.boxPoints(marker)
box = np.int0(box)
cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

cv2.putText(image, "Distance to camera: %.2fcm" % centimetres,(5,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,.5,(225,0,0))

cv2.imshow("image", image)
cv2.waitKey(0)


# OPEN SERIAL PORT TO NUCLEO BOARD
try:
	ser = serial.Serial('/dev/ttyACM0')
except SerialException:
	print('Port already open or access is denied')
	sys.exit()
	
ser.write("")

print(ser.name)



while(1):

    # Take each frame
	_, frame = cap.read()

    # Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# FIND THE OBJECTs AND MASK IT

    # define range of BALL color in HSV
    #lower_bounds = np.array([55,110,65])
    #upper_bounds = np.array([179,255,255])

    # Increment frame counter
	frameCounter = frameCounter + 1

	for (lower, upper, colorName) in color_ranges:
        # Threshold the HSV image to get only specific colors
		mask = cv2.inRange(hsv, lower, upper)

        # Additional thresholding, erode & dilate, removes noise
		mask = cv2.erode(mask, None, iterations=0)
		mask = cv2.dilate(mask, None, iterations=0)

        # Bitwise-AND mask and original image
		res = cv2.bitwise_and(frame,frame, mask= mask)

# DRAW CIRCLE & BOX

        # Find contours (A ball)
		contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
		center = None
        
		if colorName == 'ball':
			draw_circle(contours, center)
		else:
			draw_rect(frameCounter)
        
# HIT OR MISS
	timeNow = time.time()
	
	if (startTime and draw_rect.time_of_flight) != 0:
		
		if (timeNow - startTime) > timeToLaunch + draw_rect.time_of_flight:
			hit_box()
			startTime = 0


# SHOW THE FRAME


    #cv2.imshow('frame',frame)
     # These masks block out all but the ball, including the circles.
    # solution is to declare circle vars out of if statement
	cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
	cv2.imshow('view', frame)

    # Reset frame counter
	if frameCounter == 120:
		frameCounter = 0

	k = cv2.waitKey(4) & 0xFF
	if k == 27:
		break 
	if k == 108:
		ser.write(b'launch')
		print('LAUNCH!')
		startTime = time.time()			#Need to time how long it takes to move ball and it start to fire
			
cv2.destroyAllWindows()
