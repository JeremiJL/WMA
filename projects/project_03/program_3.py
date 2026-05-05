from typing import Sequence

import cv2 as cv
import sys
import screeninfo
from cv2 import Mat


def filter_matches(matches: Sequence[cv.DMatch], kp2: cv.KeyPoint) -> Sequence[Sequence[cv.DMatch]]:
    # filter matches based on distance comparison to second best match
    distance_filtered = []

    average_x = 0
    for m, n in matches:
        t_idx = m.trainIdx
        average_x += kp2[t_idx].pt[0]
        if m.distance < 0.5 * n.distance:
            distance_filtered.append([m])

    average_x /= len(matches)

    # take top X matches
    top_x = [m[0] for m in distance_filtered]
    top_x = sorted(top_x, key=lambda match: match.distance)
    top_x = top_x[:10]

    # filter matches based on the coordinates of the point
    coordinates_filtered = []

    for m in top_x:
        t_idx = m.trainIdx
        x = kp2[t_idx].pt[0]
        if average_x * 0.5 < x < average_x * 2:
            coordinates_filtered.append(x)

    filtered = [[m] for m in coordinates_filtered]
    return filtered

def mark_boundary(matches: Sequence[Sequence[cv.DMatch]], img_with_matches: Mat, query_img: Mat, kp2: cv.KeyPoint):
    xs = []
    ys =[]

    for match in matches:
        t_idx = match[0].trainIdx
        (x1, y1) = kp2[t_idx].pt
        xs.append(x1)
        ys.append(y1)

    xs.sort()
    ys.sort()

    width_shift = query_img.shape[1]

    top_left_corner = (int(xs[0] - 30) + width_shift, int(ys[0] - 30))
    bottom_right = (int(xs[len(xs) - 1] + 30 + width_shift), int(ys[len(ys) - 1] + 30))
    cv.rectangle(img_with_matches, top_left_corner, bottom_right, (0, 255, 0), 3)

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

    filtered_matches = filter_matches(matches=matches, kp2=kp2)

    # mark matches
    img_with_matches = cv.drawMatchesKnn(img_query,
                             kp1,
                             frame,
                             kp2,
                             filtered_matches,
                             None,
                             matchColor=(100, 255, 0),
                             singlePointColor=(0, 0, 255),
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # mark boundary
    mark_boundary(filtered_matches, query_img=img_query, img_with_matches=img_with_matches, kp2=kp2)
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
