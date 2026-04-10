import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("resources/tray1.jpg"))

if img is None:
    sys.exit("Could not read the image.")


converted_for_display = cv.cvtColor(img, cv.COLOR_YCrCb2BGR)
cv.imshow("Display window", converted_for_display)
cv.waitKey(0)