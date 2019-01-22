import cv2
import numpy as np
import math
import hsv_slide
import update_hsv
import object_distance

# Set capture device
cap = cv2.VideoCapture(1)

# Initialise frame counter
frameCounter = 0

# Set frame size and FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)	
cap.set(cv2.CAP_PROP_FPS, 60)

# Run initial object distance calibration script
trainKnownWidth, camFocalLength = object_distance.runScript(cap)

# Get initial HSV values
HSVLOW, HSVHIGH = hsv_slide.runScript(cap)

newHSVLOW = HSVLOW
newHSVHIGH = HSVHIGH


while(1):
	
	# Take each frame
	_, frame = cap.read()
	
	# Increment the frame count
	frameCounter = frameCounter + 1
	
	# Convert BGR to HSV
	hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
	# Apply the range on the new HSV image and create a mask
	mask = cv2.inRange(hsvImage, newHSVLOW, newHSVHIGH)
	
	# Bitwise-AND mask with the original frame. This produces an image that only shows the pixels that are in the HSV range
	res = cv2.bitwise_and(frame, frame, mask = mask)

	# Find contours 
	contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	
	if len(contours) > 0:		# If the number (length) of the contours array is more than 0. I.e any contours present
		
		# Find biggest contour
		maxCnt = max(contours, key=cv2.contourArea)
	
		for c in contours:			# Look at all contours
		
			perimeter = cv2.arcLength(c, True)      # Calulate perimeter length
			approxShape = cv2.approxPolyDP(c, 0.1 * perimeter, True)        # Approximate shape. This returns a list of points where edges connect
			
			# If the shape has 4 edges
			if len(approxShape) == 4:
				
				rect = cv2.minAreaRect(c)			# rect = ((center_x, center_y), (width,height),angle)
				points = cv2.boxPoints(rect)		        # Find four vertices of rectangle from above rect
				box = np.int0(points)				# Round the values and make them integers
				
				width = rect[1][0]
				height = rect[1][1]

				if (len(c) != len(maxCnt)):		# If current contour is not the biggest contour
					
					if ((float(width)/height)==1):  # Check if width and height are the same = Square
						cv2.putText(frame, "Square", (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0),1)	# Place text "Square"				
						cv2.drawContours(frame, [c], -1, (255, 255, 0), 2)              # Draw contour
					else:
						cv2.putText(frame, "Rectangle", (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0),1)	# Place text "Rectangle"				
						cv2.drawContours(frame, [c], -1, (255, 0, 0), 2)                # Draw contour
								
			# Shapes other than rectangles/squares		
			else:
				M = cv2.moments(c)              # Take moments of shape to find centre point
				if M["m00"] != 0:
					cX = int(M["m10"] / M["m00"])
					cY = int(M["m01"] / M["m00"])
					cv2.drawContours(frame, [c], -1, (0 ,0 ,255), 2)        # Draw contour
					cv2.putText(frame, "Other", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)    # Place text "Other"
		
		# Looking at biggest contour. Assumes biggest contour is the target
		perimeter = cv2.arcLength(maxCnt, True)         # Calculate perimeter length
		approxShape = cv2.approxPolyDP(maxCnt, 0.1 * perimeter, True)           # Approximate shape
		# If the biggest contour is a rectangle/square
		if len(approxShape) == 4:
			rect = cv2.minAreaRect(maxCnt)							# rect = ((center_x, center_y), (width,height),angle)
			points = cv2.boxPoints(rect)							# Find four vertices of rectangle from above rect
			box = np.int0(points)								# Round the values and make them integers
								
			width = rect[1][0]
			height = rect[1][1]

			# Check if the length of the height and width != 1. Therefore rectangle
			if ((float(width)/height)!=1):
				cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)		# Draw the contours of [box] onto the original frame				
				cv2.putText(frame, "Target Rectangle", (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),1)        # Place text "Target Rectangle"			
				
                                # Object distance Calculations
                                perWidth = rect[1][0]                   # Perceived width of target in pixels
                                
                                # Calculate distance of target from camera
                                disToCam = object_distance.distance_to_camera(trainKnownWidth, camFocalLength, perWidth)

                                # Put the target distance as text of frame
                                cv2.putText(frame, "Distance to camera: %.2fcm" % disToCam,(5,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,.5,(225,0,0))
		
                                # Calculate average hue, saturation and val of the detected rectangle
                                if frameCounter % 5 == 0:
                                        averageHue, averageSat, averageVal, hueWidth, satWidth, valWidth = update_hsv.runScript(cap, hsvImage, box)
	
                                        # Set new HSV values
                                        newHSVLOW = (averageHue - hueWidth, averageSat - satWidth, averageVal - valWidth)
                                        newHSVHIGH = (averageHue + hueWidth, averageSat + satWidth, averageVal + valWidth)
				
                                        print("HSV lower bounds:", newHSVLOW, "HSV upper bounds:",newHSVHIGH)	

	# Show original frame with target detection and text
	cv2.imshow('Object Detection', frame)
	# Show the resultant masked frame
	cv2.imshow('Result Masking',res)

	# Check if 60 frames have passed
	if frameCounter == 60:
		frameCounter = 0
	
	# Wait for 5ms and see if the 'ESC' or 'r' key is pressed
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break
	if k == 114:
		newHSVLOW = HSVLOW
		newHSVHIGH = HSVHIGH

# Close all windows
cv2.destroyAllWindows()
	

