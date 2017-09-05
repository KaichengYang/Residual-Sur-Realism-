# -*- coding: utf8 -*-
from PIL import Image
import residuals
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ################## ################## ##################
    # Example 1: process "./ADS.jpg"
    # open the image with PIL
    ADS_image = Image.open('./ADS.jpg')
    # initialize the ImageProcessor instance
    ADS_image_processor = residuals.ImageProcessor(ADS_image)
    # extract the coordinates of black dots from the image
    ADS_image_processor.extractCoordiantes()
    # normalize the coordinates
    ADS_image_processor.normalizeCoordinates()
    # add frame to the image so that x-coordinates and y-coordinates are
    # orthogonal
    ADS_image_processor.addFrame()
    # compare the inner product before and after adding the frame
    print "Inner product of x-coordinates and y-coordinates before adding the frame :%.2E" % sum(ADS_image_processor.normalized_xs * ADS_image_processor.normalized_ys)
    print "Inner product of x-coordinates and y-coordinates after adding the frame :%.2E" % sum(ADS_image_processor.final_xs * ADS_image_processor.final_ys)
    # plot the scatter graph of the processed data, a frame had been added to
    # the picture
    plt.scatter(ADS_image_processor.final_xs, ADS_image_processor.final_ys, s=0.2)
    plt.show()

    ################## ################## ##################
    # Example 2: process "./FA2017ADS.jpg"
    # open the image with PIL
    FA2017ADS_image = Image.open("./FA2017ADS.jpg")
    # initialize the ImageProcessor instance
    FA2017ADS_image_processor = residuals.ImageProcessor(FA2017ADS_image)
    # directly addFrame is also working
    FA2017ADS_image_processor.addFrame()
    # compare the inner product before and after adding the frame
    print "Inner product of x-coordinates and y-coordinates before adding the frame :%.2E" % sum(FA2017ADS_image_processor.normalized_xs * FA2017ADS_image_processor.normalized_ys)
    print "Inner product of x-coordinates and y-coordinates after adding the frame :%.2E" % sum(FA2017ADS_image_processor.final_xs * FA2017ADS_image_processor.final_ys)
    # plot the scatter graph of the processed data, a frame had been added to
    # the picture
    plt.scatter(FA2017ADS_image_processor.final_xs, FA2017ADS_image_processor.final_ys, s=0.2)
    plt.show()
