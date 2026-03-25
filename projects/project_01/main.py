from solutions.project1 import secret_image_processing
import sys, numpy as np, cv2 as cv
import screeninfo

screen = screeninfo.get_monitors()[0]
width, height = screen.width, screen.height

cv.namedWindow("p1", cv.WINDOW_NORMAL)
cv.moveWindow("p1", screen.x - 1, screen.y - 1)
cv.setWindowProperty("p1", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

capture = cv.VideoCapture(2, cv.CAP_ANY)

if not capture.isOpened():
    print("Failed to initialise camera capture. Check camera connection")
    sys.exit(1)

last_frame = None
while True:
    _, frame = capture.read()

    # implement this
    frame = secret_image_processing(frame)

    frame = frame if frame is not None else last_frame
    last_frame = frame

    cv.imshow("p1", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
