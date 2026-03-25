from typing import Sequence, List

import cv2 as cv
import numpy as np
from pathlib import Path
from cv2 import Mat

IMAGE_PATH = "resources/red_ball.jpg"
EXIT_KEY = 0

def import_image(image_path) -> Mat:
    if not Path(image_path).is_file():
        raise ValueError("Provided file doesn't exist under specified path.")

    image = cv.imread(image_path)
    assert image is not None, "Provided file is not a proper image."
    return image

def convert_to_hsv_color_space(image) -> Mat:
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    return hsv

def show(window_name, image):
    cv.imshow(window_name, image)

def apply_mask(image) -> Mat:
    lower_red = np.array([0, 200, 0], np.uint8)

    upper_red = np.array([30, int(60/100) * 255, int(50/100) * 255], np.uint8)
    mask0 = cv.inRange(image, lower_red, upper_red)

    lower_red = np.array([150, 150, 0], np.uint8)
    upper_red = np.array([179, 255, 255], np.uint8)
    mask1 = cv.inRange(image, lower_red, upper_red)

    combined_mask = mask0 | mask1

    return combined_mask

def remove_noise(image) -> Mat:
    kernel = np.ones((6,6), dtype=int)
    image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    # image = cv.GaussianBlur(image, (3, 3), 0)
    return image

def find_greatest_contour(contours_sequence: Sequence[Mat]) -> Mat:
    contours_list = list(contours_sequence)
    contours_list.sort(key = cv.contourArea, reverse=True)
    return contours_list[0]

def apply_contours(original_image, image_with_mask) -> tuple[Mat,Mat]:
    contours_list, hierarchy = cv.findContours(image_with_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    main_contour = find_greatest_contour(contours_list)
    cv.drawContours(original_image, [main_contour], 0, (255,255,255), 3)
    return original_image, main_contour

def compute_center_of_gravity(contour):
    m: dict[str, float] = cv.moments(contour)
    area = m.get('m00')
    m10 = m.get('m10')
    m01 = m.get('m01')
    center_of_gravity: tuple[int, int] = (int(m10 / area), int(m01 / area))
    return center_of_gravity

def apply_text_and_centre_point(image, main_contour):
    center_of_gravity = compute_center_of_gravity(contour=main_contour)
    cv.circle(image, center_of_gravity, 2, (255, 255, 255), -1)
    cv.putText(image, 'red ball', (center_of_gravity[0], center_of_gravity[1]-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def run():
    original_image = import_image(image_path=IMAGE_PATH)
    print(original_image)
    image_in_hsv = convert_to_hsv_color_space(image=original_image)
    image_with_mask = apply_mask(image_in_hsv)
    image_without_noise = remove_noise(image_with_mask)
    image_with_contours, contour = apply_contours(image_with_mask=image_without_noise, original_image=original_image)
    apply_text_and_centre_point(image=original_image, main_contour=contour)

    show(window_name="original image with contour and text",image=original_image)
    show(window_name="image in csv",image=image_in_hsv)
    show(window_name="image with applied mask",image=image_with_mask)
    show(window_name="image without noise",image=image_without_noise)
    show(window_name="image with contours",image=image_with_contours)
    cv.waitKey(EXIT_KEY)

run()
