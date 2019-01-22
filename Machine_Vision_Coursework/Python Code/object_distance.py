import cv2
import numpy as np

# IMAGE DISTANCE CALIBRATION DATA
KNOWN_DISTANCE = 60
KNOWN_WIDTH = 25.9
# ************ THIS PATH WILL NEED TO BE CHANGED WHEN RUN ON ANOTHER PC ************ #
IMAGE_PATH = "/home/elliottwhite/ROCO318/MachineVision/CourseworkFinal/60cm_image_2.png"

# FUNCTION FOR INTIAL CALIBRATION TO FIND THE TARGET DISTANCE
def find_target(image):

        # Convert image to HSV colourspace
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	
        # Create mask from previously determined range
	mask = cv2.inRange(hsv, (21,46,117),(28,255,255))	

	# Find contours in image
	contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]

        # Find max contour and find bounding box
	c = max(contours, key=cv2.contourArea)
	rect = cv2.minAreaRect(c)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	# Draw the box contour on the image
	cv2.drawContours(image, [box], 0, (0, 255, 255), 2)
 
	# Compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)


# FUNCTION TO CALCULATE DISTANCE OF TARGET FROM CAMERA
def distance_to_camera(knownWidth, focalLength, perWidth):

        # Check if perceived width != 0
	if perWidth != 0:
		# Compute and return the distance from the target to the camera
		return (knownWidth * focalLength) / perWidth
	else:
		return 0
		
		

def runScript(cap):
	# CALIBRATE CAMERA
	image = cv2.imread(IMAGE_PATH)
	# Run find target script, given the image path
	targetAreaRect = find_target(image)
	# Calculate perceived witdh
	perWidth = targetAreaRect[1][0]
	# Calculate focal length
	focalLength = (perWidth * KNOWN_DISTANCE) / KNOWN_WIDTH

        # Calculate distance target is from camera
	centimetres = distance_to_camera(KNOWN_WIDTH, focalLength, perWidth)

        # Create box points from bounding rectangle
	box = cv2.boxPoints(targetAreaRect)
	box = np.int0(box)
	# Draw box contour on image
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

        # Put distance to target as text on image
	cv2.putText(image, "Distance to camera: %.2fcm" % centimetres,(5,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,.5,(225,0,0))

        # Show resultant image
	cv2.imshow("image", image)

	# Wait for any key to be pressed
	cv2.waitKey(0)

	# Return known_width and focalLength back to main.py
	return KNOWN_WIDTH, focalLength
