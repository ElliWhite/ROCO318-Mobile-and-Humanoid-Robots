import cv2
import numpy as np

def nothing(x):
	pass

# Create empty arrays
HSVLOW=np.zeros(3)
HSVHIGH=np.zeros(3)

# Main script
def runScript(cap):

        # Create a window
	cv2.namedWindow('HSV Masking')

        # Here for easy assignments
	hh='Hue High'
	hl='Hue Low'
	sh='Saturation High'
	sl='Saturation Low'
	vh='Value High'
	vl='Value Low'

        # Create trackbars on frames and set boundaries
	cv2.createTrackbar(hl, 'HSV Masking',0,179,nothing)
	cv2.createTrackbar(hh, 'HSV Masking',179,179,nothing)
	cv2.createTrackbar(sl, 'HSV Masking',0,255,nothing)
	cv2.createTrackbar(sh, 'HSV Masking',255,255,nothing)
	cv2.createTrackbar(vl, 'HSV Masking',0,255,nothing)
	cv2.createTrackbar(vh, 'HSV Masking',255,255,nothing)

	while(1):

		# Read in a frame from capture device
		_,frame=cap.read()

		# Add some Gaussian Blur
		frame=cv2.GaussianBlur(frame,(5,5),0)
		
		#convert to HSV from BGR
		hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		#read trackbar positions for all
		hul=cv2.getTrackbarPos(hl, 'HSV Masking')
		huh=cv2.getTrackbarPos(hh, 'HSV Masking')
		sal=cv2.getTrackbarPos(sl, 'HSV Masking')
		sah=cv2.getTrackbarPos(sh, 'HSV Masking')
		val=cv2.getTrackbarPos(vl, 'HSV Masking')
		vah=cv2.getTrackbarPos(vh, 'HSV Masking')
		
		#make array for final values
		HSVLOW=(hul,sal,val)
		HSVHIGH=(huh,sah,vah)

		# Create a mask from the range
		mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)
		# Add the mask onto the frame
		res = cv2.bitwise_and(frame,frame, mask = mask)

                # Show resultant masked frame
		cv2.imshow('HSV Masking', res)

		# Wait 5ms and see if 'ESC' key was pressed
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
			break

        # Close all windows and return new arrays
	cv2.destroyAllWindows()
	return HSVLOW, HSVHIGH
