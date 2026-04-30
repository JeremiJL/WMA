import cv2 as cv
import sys
import screeninfo

def process_matching(frame: cv.Mat, img_query: cv.Mat) -> cv.Mat:
    # cast to gray scale
    query_img_gray = cv.cvtColor(img_query, cv.COLOR_BGR2GRAY)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(query_img_gray, None)
    kp2, des2 = sift.detectAndCompute(frame_gray, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # filter poor matches
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])

    # take top X matches
    good_matches = [m[0] for m in good]
    good_matches = sorted(good_matches, key=lambda match: match.distance)
    good_matches = good_matches[:10]

    good = [[m] for m in good_matches]

    # mark matches
    img_with_matches = cv.drawMatchesKnn(img_query,
                             kp1,
                             frame,
                             kp2,
                             good,
                             None,
                             matchColor=(100, 255, 0),
                             singlePointColor=(0, 0, 255),
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_with_matches



def display():
    screen = screeninfo.get_monitors()[0]
    cv.namedWindow("p1", cv.WINDOW_NORMAL)
    cv.moveWindow("p1", screen.x - 1, screen.y - 1)
    cv.setWindowProperty("p1", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    capture = cv.VideoCapture("resources/video_3_train.mp4", cv.CAP_ANY)

    if not capture.isOpened():
        print("Failed to initialise camera capture. Check camera connection")
        sys.exit(1)

    query_img = cv.imread(cv.samples.findFile("resources/photo_3_query.jpg"))

    last_frame = None
    while True:
        _, frame = capture.read()

        frame = process_matching(frame, query_img)

        frame = frame if frame is not None else last_frame
        last_frame = frame

        cv.imshow("p1", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
  display()
