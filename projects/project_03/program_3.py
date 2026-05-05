from typing import Sequence

import cv2 as cv
import sys
import screeninfo
from cv2 import Mat


def filter_matches(matches: Sequence[Sequence[cv.DMatch]], kp2: Sequence[cv.KeyPoint]) -> Sequence[Sequence[cv.DMatch]]:

    filtered = []

    # filter matches based on distance comparison to second best match
    distance_filtered = []

    if not matches:
        return []

    for pair in matches:
        if len(pair) < 2:
            continue

        m, n = pair
        if m.distance < 0.5 * n.distance:
            distance_filtered.append(m)

    if not distance_filtered:
        return []

    # update filtered list
    filtered = distance_filtered

    # take top X matches
    top_A_matches = []

    top_A_matches = sorted(filtered, key=lambda match: match.distance)
    top_A_matches = top_A_matches[:10]

    # update filtered list
    filtered = top_A_matches

    # filter matches based on the coordinates of the point
    coordinates_filtered = []

    average_x = 0
    average_y = 0
    for m in filtered:
        t_idx = m.trainIdx
        (x, y) = kp2[t_idx].pt
        average_x += x
        average_y += y

    average_x /= len(filtered)
    average_y /= len(filtered)

    for m in filtered:
        t_idx = m.trainIdx
        (x,y) = kp2[t_idx].pt
        if (x < average_x * 1.5) and (y > average_y * 0.7):
            coordinates_filtered.append(m)

    # update filtered list
    filtered = coordinates_filtered

    # format filtered list
    filtered = [[m] for m in filtered]
    return filtered

def mark_boundary(matches: Sequence[Sequence[cv.DMatch]], img_with_matches: Mat, query_img: Mat, kp2: Sequence[cv.KeyPoint]):
    if not matches:
        return

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
    if frame is None or img_query is None:
        return frame

    # cast to gray scale
    query_img_gray = cv.cvtColor(img_query, cv.COLOR_BGR2GRAY)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(query_img_gray, None)
    kp2, des2 = sift.detectAndCompute(frame_gray, None)

    if des1 is None or des2 is None or len(kp2) == 0:
        return frame

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
        ok, frame = capture.read()
        if not ok:
            break

        frame = process_matching(frame, query_img)

        frame = frame if frame is not None else last_frame
        last_frame = frame

        cv.imshow("p1", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
  display()
