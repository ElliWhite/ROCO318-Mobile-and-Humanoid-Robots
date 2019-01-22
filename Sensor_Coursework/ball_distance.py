import cv2
import numpy as np


cap = cv2.VideoCapture(1)


#ESTIMATE focal length
area = 40.0
known_dist = 10.0 #10cm from camera
known_width = 3.0 #Ball is 3cm diameter

#VF0800 FOV
fov = 77


def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth

focal_length = (area * known_dist)/known_width #change area to width in pixels
print('focal length',focal_length)

distance = distance_to_camera(known_width, focal_length, area) #area is actually width in pixels
print('distance', distance)


while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# FIND THE BALL AND MASK IT

    # define range of color in HSV
    lower_bounds = np.array([95,130,85])
    upper_bounds = np.array([179,255,255])

    #find_ball(hsv, lower_bounds, upper_bounds, mask)

    # Threshold the HSV image to get only specific colors
    mask = cv2.inRange(hsv, lower_bounds, upper_bounds)

    # Additional thresholding, erode & dilate, removes noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

# AREA OF BALL
    # Find contours (A ball)
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        #(x,y),radius = cv2.minEnclosingCircle(c)
        print(radius)
        # Print X & Y of ball
        #print(c)


# SHOW THE FRAME

    cv2.imshow('frame',frame)
    # These masks block out all but the ball, including the circles.
    # solution is to declare circle vars out of if statement
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    #cv2.imshow('yay', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
