#!/usr/bin/env python

"""Crop and resize eye of fisheye camera image

input:
- raw image
output:
- cropped and resized image, centered around optical center_
"""

# For Python2/3 compatibility
from __future__ import print_function
from __future__ import division

__author__ = 'Eric Dortmans'
__email__ = 'eric.dortmans@gmail.com'

import argparse
import signal
import sys
import os
import numpy as np
import cv2


# Colors (bgr)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)


def find_eye(image):
    """find eye of image

    """
    height, width = image.shape[:2]

    # First estimate
    cx, cy, radius = width // 2, height // 2, height // 2 + height // 4

    # Better estimate
    cnt = find_largest_contour(image)
    cnt = cv2.convexHull(cnt)
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    cx, cy, radius = int(cx), int(cy), int(radius)

    return cx, cy, radius


def find_largest_contour(image, threshold = 50):
    """find largest contour in image
    """
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    _, cnts, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    return cnt


def crop(image, x, y, w, h, margin=0):
    """Crop image

    https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python

    x, y: corner of region of interest (roi)
    w, h: width and height of roi
    """

    roi = image[y - margin : y + h + margin, x - margin : x + w + margin]

    return roi


def resize(image, size):
    """Resize image

    size: target size of image
    """
    (width, height) = (size, size)
    resized_image = cv2.resize(image, (width,height), interpolation=cv2.INTER_CUBIC)

    return resized_image


def mask_eye(image, cx, cy, radius):
    """mask image to focus on the eye

    cx, cy: center of eye
    radius: radius of eye
    """

    mask = np.zeros(image.shape[:2], np.uint8)
    mask = cv2.circle(mask, (cx, cy), radius, WHITE, -1)
    image_masked = cv2.bitwise_and(image, image, mask=mask)

    return image_masked


def reshape(image, shape=(480,480)):
    """ Reshape image.

    """
    h, w = image.shape[:2]
    H, W = shape[:2]

    top, bottom, left, right = 0, 0, 0, 0   # paddings around the image.
    if H > h:
        top = int((H - h)/2.0)
        bottom = H-h-top
    if W > w:
        left = int((W - w)/2.0)
        right = W-w-left

    # Copy the image into the middle of the destination image
    reshaped = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = 0)

    return reshaped


def eye_crop_resize(image, image_out_size=300, reduction=0.9, display=False):
    """ Crop and resize square patch containing the eye of the image

    image: fisheye camera image (any size)
    size: size of output image (number of pixels)
    reduction: reduction factor (0.0 - 1.0) to cutoff some of the border of the eye

    image_out: output image (size x size)
    """

    height, width = image.shape[:2]
    
    if display:
        display_image = image.copy()

    # find eye
    cx, cy, radius = find_eye(image)

    if display:
        #print("cx, cy, radius",cx, cy, radius)
        cv2.circle(display_image, (cx, cy), radius, BLUE, 3)
        cv2.circle(display_image, (cx, cy), 10, RED, -1)

    # Reduce eye to remove border
    radius = int(radius * reduction)

    if display:
        cv2.circle(display_image, (cx, cy), radius, RED, 3)    

    # remove everything outside the reduced eye circle
    image = mask_eye(image, cx, cy, radius)



    # cutout the rectangle with the eye
    x, y = max(0, cx - radius), max(0, cy - radius)
    w, h = min(width, cx - x + radius), min(height, cy - y + radius)
    cropped_rectangle = crop(image, x, y, w, h)

    # make it a square image
    size = max(cropped_rectangle.shape[0], cropped_rectangle.shape[1])
    image = reshape(cropped_rectangle, (size, size))

    # resize it to required output size
    image_out = resize(image, image_out_size)

    if display:
    	cv2.imshow("image", display_image)
    	cv2.waitKey(0)

    return image_out


def main():

    # construct the argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path", help="/path/to/image or /path/to/image_directory")
    ap.add_argument("-o", "--image_output_path", help="/path/to/output_image_directory")
    ap.add_argument("-s", "--image_output_size", type=int, default=300, help="Size of output image")
    ap.add_argument("-r", "--eye_reduction", type=float, default=0.9, help="Eye size reduction factor")
    ap.add_argument("-d", "--display", action='store_true', default=False, help="Display image")
    args = ap.parse_args()

    if args.image_output_path:
        # Create output dir if it does not exist
        if os.path.exists(args.image_output_path):
            pass
        else:
        	os.mkdir(args.image_output_path)

    # Generate list of images to be processed
    if os.path.isdir(args.image_path):
        images = []
        for root, d_names, f_names in os.walk(args.image_path):
            for f in f_names:
                images.append(os.path.join(root, f))
    else:
        images = [args.image_path]

    # process each image
    for image in images:
        file_dir = os.path.dirname(image)
        file_base = os.path.basename(image)
        file_name, file_extension = os.path.splitext(file_base)

        # load the image
        image = cv2.imread(image)

        # crop, and resize the image
        image_out = eye_crop_resize(image, args.image_output_size, args.eye_reduction, args.display)

        # write the resulting image to a file
        if args.image_output_path:
            out_dir = args.image_output_path
        else:
            out_dir = file_dir

        file = file_name + '_resized' + file_extension
        path = os.path.join(out_dir, file)
        print("Created:", path)
        cv2.imwrite(path, image_out)


if __name__ == "__main__":

	try:
		main()
	except KeyboardInterrupt:
		pass
	finally:
		cv2.destroyAllWindows()



    
