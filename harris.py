"""harris.py"""
__author__ = "Gurveer Dhindsa"

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""
Main function
"""
def main():
    # Load the image
    image = cv.imread("box_in_scene.png")

    # Display the original image (colour)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Original image")
    plt.show()

    # Parameters
    block = 5
    aperature = 5

    # Define the window (which I later reference for the slider)
    window = 'Harris Corner Detector'

    # Define threshold values (slider)
    thresholdValue = 50
    maxThresholdValue = 100

    # Convert image to grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.cornerMinEigenVal(image, block, aperature)

    dstMin = np.amin(dst)
    dstMax = np.amax(dst)

    def detect_corners(val):
        # Make a copy of the image each time threshold slider changes
        image2 = np.copy(image)
        thresholdValue = max(val, 1)
        # Iterate the pixels in the image
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                # Non-maximum supression
                if dst[x,y] > dstMin+(dstMax-dstMin)*thresholdValue/maxThresholdValue:
                    # Draw the circle
                    cv.circle(image2, (y, x), 3, 255, 1)
        cv.imshow(window, image2)


    # Instantiate window
    cv.namedWindow(window)
    
    # Instantiate threshold slider
    cv.createTrackbar('Threshold:', window, thresholdValue, maxThresholdValue, detect_corners)
    detect_corners(thresholdValue)
    cv.waitKey()

"""
Determines if image is grayscale
(Helper function)

Args:
    image - the image being examined

Returns:
    A boolean value whether or not the image is already grayscale
"""
def isImageGrayscale(image):
    image = Image.fromarray(image) # Convert array back to image
    width, height = image.size # Grab the image dimensins

    # Iterate the image pixel by pixel...
    for w in range(width):
        for h in range(height):
            # Grab the RGB values
            r, g, b = image.getpixel((w, h))
            # If they are individually different, then we know it CANNOT be a grayscale image
            if r != g != b: 
                return False
    # If we got here, then the image is grayscale
    return True

if __name__ == '__main__':
   main()