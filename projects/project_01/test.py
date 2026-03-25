# pip install opencv-python
import cv2 as cv # <--- "cv" specifically
import numpy as np

robopic = cv.imread("resources/red_ball.jpg")

# img = cv.cvtColor(image, colour space)
# BGR2HSV -> BGR to hue sat val
# blue -> (255, 0, 0)-
hsv = cv.cvtColor(robopic, cv.COLOR_BGR2HSV)

lower_red = np.array([0, 150, 0], np.uint8) 
upper_red = np.array([10, 255, 255], np.uint8) 
# (image, upper bound, lower bound)
mask0 = cv.inRange(hsv, lower_red, upper_red)

lower_red = np.array([165, 150, 0], np.uint8) 
upper_red = np.array([179, 255, 255], np.uint8) 
# (image, upper bound, lower bound)
mask1 = cv.inRange(hsv, lower_red, upper_red)

combined_mask = mask0 | mask1

# putText(image, text, , font, fontsize, colour, thickness)
col = (255, 0, 0)
cv.putText(robopic, 'red ball', (25, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

cv.imshow("robot", robopic)
cv.imshow("hsv", hsv)
cv.imshow("mask0", combined_mask)
cv.waitKey(0)
