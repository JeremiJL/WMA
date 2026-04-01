import sys, numpy as np, cv2 as cv
import screeninfo
from program_2 import process_contours

screen = screeninfo.get_monitors()[0]
width, height = screen.width, screen.height

cv.namedWindow("p1", cv.WINDOW_NORMAL)
cv.moveWindow("p1", screen.x - 1, screen.y - 1)
cv.setWindowProperty("p1", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

capture = cv.VideoCapture("resources/rgb_ball_720.mp4", cv.CAP_ANY)

if not capture.isOpened():
    print("Failed to initialise camera capture. Check camera connection")
    sys.exit(1)

last_frame = None
while True:
    _, frame = capture.read()

    frame = process_contours(frame)

    frame = frame if frame is not None else last_frame
    last_frame = frame

    cv.imshow("p1", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
