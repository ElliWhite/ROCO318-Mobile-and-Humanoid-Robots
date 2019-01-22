
import cv2
import numpy as np

cap = cv2.VideoCapture(1)

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
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 1:
            #cv2.circle(img, center, radius, color, thickness=1, lineType=8, shift=0)
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)


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



# SHOW THE FRAME

    cv2.imshow('frame',frame)
    # These masks block out all but the ball, including the circles.
    # solution is to declare circle vars out of if statement
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.imshow('yay', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
