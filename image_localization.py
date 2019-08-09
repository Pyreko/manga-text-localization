#!/usr/bin/env python3

import imutils
import cv2
import numpy as np
import time
import os
import argparse
from pathlib import Path
import pytesseract
from PIL import Image
import sys


def image_localization(img_path, east_net_path, should_draw_rect, debug):
    start = time.time()
    image = cv2.imread(img_path)
    return_image = add_text_boxes(image, east_net_path,
                                  should_draw_rect, debug)
    end = time.time()

    print("Time taken for one page: {} seconds.".format(end - start))
    return return_image


def add_text_boxes(original_image, east_net_path,
                   should_draw_rect, debug):
    """Adds bounding boxes to the original image."""

    final_image = original_image.copy()
    (old_height, old_width) = final_image.shape[:2]

    processed_width = ((int)(old_width / 32) + 1) * 32 * 2
    processed_height = ((int)(old_height / 32) + 1) * 32 * 2
    processed_image = cv2.resize(final_image, (processed_width, processed_height), interpolation=cv2.INTER_CUBIC)
    processed_image = cv2.medianBlur(processed_image, 5)

    (original_height, original_width) = final_image.shape[:2]
    height_ratio = original_height / float(processed_height)
    width_ratio = original_width / float(processed_width)

    layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # Detect
    east_text_detector = cv2.dnn.readNet(east_net_path)
    blob = cv2.dnn.blobFromImage(processed_image,
                                 1.0, (processed_width, processed_height),
                                 (123.68, 116.78, 103.94),
                                 swapRB=True,
                                 crop=False)
    east_text_detector.setInput(blob)
    (scores, geometry) = east_text_detector.forward(layer_names)

    # Bounding boxes
    (num_rows, num_cols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, num_rows):
        scores_data = scores[0, 0, y]
        x_data_0 = geometry[0, 0, y]
        x_data_1 = geometry[0, 1, y]
        x_data_2 = geometry[0, 2, y]
        x_data_3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(0, num_cols):
            if scores_data[x] < 0.9:
                continue

            # Fixes offset
            (offset_x, offset_y) = (x * 4.0, y * 4.0)

            cos = np.cos(angles_data[x])
            sin = np.sin(angles_data[x])

            bounding_box_height = x_data_0[x] + x_data_2[x]
            bounding_box_width = x_data_1[x] + x_data_3[x]

            start_x = int(offset_x + (cos * x_data_1[x]) +
                          (sin * x_data_2[x])) - bounding_box_width
            start_y = int(offset_y - (sin * x_data_1[x]) +
                          (cos * x_data_2[x])) - bounding_box_height

            # Rescale
            start_x = int(start_x * width_ratio)
            start_y = int(start_y * height_ratio)
            bounding_box_width = int(bounding_box_width * width_ratio)
            bounding_box_height = int(bounding_box_height * height_ratio)

            rects.append([
                start_x, start_y,
                int(bounding_box_width),
                int(bounding_box_height)
            ])
            confidences.append(float(scores_data[x]))

    idxes = cv2.dnn.NMSBoxes(rects, confidences, 0.9, 0.4)

    # TODO: Rotated boxes

    final_rects = []
    used_rect = []
    PIXEL_OFFSET = 20  # This is hardcoded and temp
    for i in idxes.flatten():
        start_x = int(rects[i][0] - (PIXEL_OFFSET / 2))
        start_y = int(rects[i][1] - PIXEL_OFFSET)
        end_x = int(start_x + rects[i][2] + PIXEL_OFFSET)
        end_y = int(start_y + rects[i][3] + PIXEL_OFFSET * 2)
        final_rects.append((start_x, start_y, end_x, end_y))
        used_rect.append(False)

    actual_final_rects = []

    for i, _ in enumerate(final_rects):
        if used_rect[i]:
            continue
        for j, _ in enumerate(final_rects):
            if i == j or used_rect[j]:
                continue

            if is_rect_intersection(final_rects[i], final_rects[j]):
                new_rect = rect_union(final_rects[i], final_rects[j])
                used_rect[i] = True
                used_rect[j] = True
                final_rects.append(new_rect)
                used_rect.append(False)
                break

    for i, _ in enumerate(used_rect):
        if not used_rect[i]:
            (x1, y1, x2, y2) = final_rects[i]
            actual_final_rects.append((x1 + int(PIXEL_OFFSET / 4), y1 + int(PIXEL_OFFSET / 2), x2 - int(PIXEL_OFFSET / 4), y2 - int(PIXEL_OFFSET / 2)))

    process_tesseract(final_image, actual_final_rects, debug)

    if should_draw_rect:
        for (start_x, start_y, end_x, end_y) in actual_final_rects:
            cv2.rectangle(final_image, (start_x, start_y), (end_x, end_y),
                          (255, 0, 255), 2)
        return final_image
    else:
        return None


def rect_union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[2], b[2]) - x
    h = max(a[3], b[3]) - y
    return (x, y, x + w, y + h)


def is_rect_intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[2], b[2]) - x
    h = min(a[3], b[3]) - y
    if w <= 0 or h <= 0: return False
    return True

def process_tesseract(image, rects, debug):

    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)

    for (start_x, start_y, end_x, end_y) in rects:
        RATIO = 5
        crop = bw_image[start_y:end_y, start_x:end_x]
        (cropped_height, cropped_width) = crop.shape[:2]

        if cropped_height <= 0 or cropped_width <= 0:
            continue

        final_cropped_processed_image = cv2.resize(crop, None, fx=RATIO, fy=RATIO, interpolation=cv2.INTER_CUBIC)
        #final_cropped_processed_image = cv2.medianBlur(final_cropped_processed_image, 5)
        (_, final_cropped_processed_image) = cv2.threshold(final_cropped_processed_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Determine background colour and invert if needed!
        mean_bgr = np.round(np.mean(final_cropped_processed_image[:], axis=(0,1))).astype(np.uint8)
        is_bg_color_black = np.all(mean_bgr < (60, 60, 60))

        if not is_bg_color_black:
            final_cropped_processed_image = cv2.bitwise_not(final_cropped_processed_image)

        (contours, _) = cv2.findContours(final_cropped_processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        final_cropped_processed_image = cv2.bitwise_not(final_cropped_processed_image)
        final_cropped_processed_image = cv2.medianBlur(final_cropped_processed_image, 5)

        # Contours
        tmp = cv2.cvtColor(final_cropped_processed_image, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            x1, y1, w, h = cv2.boundingRect(contour)
            x2 = x1 + w
            y2 = y1 + h
            cv2.rectangle(tmp, (x1, y1), (x2, y2), (255, 255, 0), 1);

        # Form image for Tesseract
        cropped_img = Image.fromarray(final_cropped_processed_image)
        txt = pytesseract.image_to_string(cropped_img, lang="eng", config="--psm 4 --oem 1")

        if txt.strip():
            print(txt)
            if debug:
                cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
                cv2.imshow('Image', tmp)
                cv2.waitKey()
        


def localize_directory(directory_path, east_net_path, show_bounds, debug):
    for subdir, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = subdir + os.sep + file
            if file.endswith('.png') or file.endswith('.jpg'):
                final_image = image_localization(file_path, east_net_path,
                                                 show_bounds, debug)

                if show_bounds and final_image is not None:
                    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
                    cv2.imshow('Image', final_image)
                    cv2.waitKey()

def main():
    parser = argparse.ArgumentParser(description='OCR\'s what it expects to be a manga page and tries to get the text.')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursively OCR the path you pass.  The path now expects a directory.  Defaults to False.')
    parser.add_argument('-m', '--path_to_model', type=str, default='./frozen_east_text_detection.pb', help='Path to the EAST model.  Defaults to the one included in the git repo.')
    parser.add_argument('-i', '--image_path', type=str, help='Path to the image or directory (if -r is used).')
    parser.add_argument('-s', '--show_bounds', action='store_true', help='Will show a popup image of the bounded image.  Defaults to False.')
    parser.add_argument('-d', '--debug', action='store_true', help='Show debugging tools, like showing cropped and processed images.  Defaults to False.')
    parser.add_argument('-t', '--time', action='store_true', help='Will time how long it takes to process an image.  Defaults to False.')
    parser.add_argument('-o', '--output', type=str, help='Where to output the JSON file containing the OCR\'d text.')

    args = vars(parser.parse_args())

    if len(sys.argv) < 1: # TODO change to 2
        return

    if args["recursive"]:
        localize_directory(args["image_path"], args["path_to_model"], args["show_bounds"], args["debug"])
    else:
        final_image = image_localization(args["image_path"], args["path_to_model"], args["show_bounds"], args["debug"])

    if args["show_bounds"] and final_image is not None:
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Image', final_image)
        cv2.waitKey()

if __name__ == "__main__":
    main()
