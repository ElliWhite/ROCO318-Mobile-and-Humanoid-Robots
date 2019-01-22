import cv2
import numpy as np

def nothing(x):
	pass

# Main script
def runScript(cap, hsvImage, box):

	# Declare and initialise variables
	totalHue = 0
	totalSat = 0
	totalVal = 0
	maxHue = 0
	maxSat = 0
	maxVal = 0
	minHue = 179
	minSat = 255
	minVal = 255
	
	averageHue = 0
	averageSat = 0
	averageVal = 0
	
	differenceHue = 0
	differenceSat = 0
	differenceVal = 0

	# Add Gaussian Blur to HSV frame
	blurHSV = cv2.GaussianBlur(hsvImage, (5,5), 0)
	
	# Create empty image
	EmptyImg = np.zeros_like(blurHSV)
	
	# Draw the contours of the detected box. This if filled in the with thickness=-1 parameter
	cv2.drawContours(EmptyImg, [box], -1, 255, -1)
	
	# Find all the pixels in the image where that correspond to the set colour from the draw contours
	xyPoints = np.where(EmptyImg == 255)
	
	# Add the hsv values to a list for all found points ( x , y )
	hsv_intensities = blurHSV[xyPoints[0],xyPoints[1]]

        # Loop through all pixels and find HSV intensities
	for i in range (0, len(hsv_intensities)):
                # Add HSV intensities up
		totalHue = totalHue + hsv_intensities[i][0]
		totalSat = totalSat + hsv_intensities[i][1]
		totalVal = totalVal + hsv_intensities[i][2]

		# Find new min and max HSV intensities
		if hsv_intensities[i][0] < minHue:
			minHue = hsv_intensities[i][0]
		elif hsv_intensities[i][0] > maxHue:
			maxHue = hsv_intensities[i][0]
			
		if hsv_intensities[i][1] < minSat:
			minSat = hsv_intensities[i][1]
		elif hsv_intensities[i][1] > maxSat:
			maxSat = hsv_intensities[i][1]
			
		if hsv_intensities[i][2] < minVal:
			minVal = hsv_intensities[i][2]
		elif hsv_intensities[i][2] > maxVal:
			maxVal = hsv_intensities[i][2]
				
	# Calculate new averages	
	if len(hsv_intensities) != 0:
		averageHue = totalHue / len(hsv_intensities)
		averageSat = totalSat / len(hsv_intensities)
		averageVal = totalVal / len(hsv_intensities)

	# Calculate ranges
	differenceHue = maxHue - minHue
	differenceSat = maxSat - minSat
	differenceVal = maxVal - minVal
	
	# Create a boundary, 33% of the range
	hueWidth = differenceHue / 3
	satWidth = differenceSat / 3
	valWidth = differenceVal / 3

        # Show image with filled in contour drawn onto it
	cv2.imshow("Empty image", EmptyImg)

        # Return average values and new boundary widths to main.py
	return averageHue, averageSat, averageVal, hueWidth, satWidth, valWidth
