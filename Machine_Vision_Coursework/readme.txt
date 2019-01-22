********************************************
ROCO318 Machine Vision
Contour-Based OpenCV application

Author: Elliott White
Date: 11/01/2018
Email: elliott.white@students.plymouth.ac.uk
********************************************

1. The capture device in main.py is set to cv2.VideoCapture(1). This will need to be indexed to (0) if only one webcam is plugged in to PC. 
	Development was done with an external webcam on a laptop that had an integrated webcam so the index was (1).
2. The image file path in object_distance.py must be changed for the code to run. The path currently points to a place on the PC the algorithm
	was developed on, so will need to be changed when run on another pc. The picture it points to will be included in this folder.
