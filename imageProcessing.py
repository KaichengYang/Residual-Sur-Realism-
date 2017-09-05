# -*- coding: utf8 -*-
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class ImageProcessor(object):
    def __init__(self, image, threshold=128):
        self.image = image
        self.gray_image = self.image.convert("L")
        self.image_array = np.asarray(self.gray_image).copy()

        if threshold < 0 or threshold > 255:
            print "Invalid threshold, try again with 0 <= threshold <=255."
        self.threshold = threshold

        self.image_xs = None
        self.image_ys = None

        self.normalized_xs = None
        self.normalized_ys = None

        self.frame_x_max = None
        self.frame_x_min = None
        self.frame_y_max = None
        self.frame_y_min = None

        self.extractCoordiantes()
        self.normalizeCoordinates()

    def showImage(self):
        self.image.show()

    def showImageGray(self):
        self.gray_image.show()

    def extractCoordiantes(self):
        xs, ys = np.where(self.image_array < self.threshold)
        self.image_xs = ys
        self.image_ys = -xs
        return self.image_xs, self.image_ys

    def drawImageArray(self):
        if self.image_xs is None:
            self.extractCoordiantes()
        plt.scatter(self.image_xs, self.image_ys, s=0.5)
        plt.show()

    def drawImageArrayNormalized(self):
        if self.normalized_xs is None:
            self.normalizeCoordinates()
        plt.scatter(self.normalized_xs, self.normalized_ys, s=0.5)
        plt.show()

    def normalizeCoordinates(self):
        if self.image_xs is None:
            self.extractCoordiantes()
        image_xs_mean = self.image_xs.mean()
        self.normalized_xs = (self.image_xs - image_xs_mean) / abs(image_xs_mean)

        if self.image_ys is None:
            self.extractCoordiantes()
        image_ys_mean = self.image_ys.mean()
        self.normalized_ys = (self.image_ys - image_ys_mean) / abs(image_ys_mean)

    def addFrame(self, num_of_points=100):
        self.num_of_points = num_of_points + 1
        self.intervals = np.linspace(0, 1, self.num_of_points)

        if self.normalized_xs is None:
            self.normalizeCoordinates()

        xmax = max(abs(max(ip.normalized_xs)), abs(min(ip.normalized_xs)))
        ymax = max(abs(max(ip.normalized_ys)), abs(min(ip.normalized_ys)))
        self.frame_x_max = xmax * 1.1
        self.frame_x_min = -self.frame_x_max
        self.frame_y_max = ymax * 1.1
        self.frame_y_min = -self.frame_y_max

        self.findAlpha()

    def findAlpha(self, verbose=False):
        original_inner_product = sum(self.normalized_xs * self.normalized_ys)
        alpha = 1
        interval = 1
        while(interval > 1e-10):
            inner_product, _, _ = self.addFrameAccordingAlpha(alpha + interval)
            if verbose:
                print inner_product, alpha, interval
            if original_inner_product * inner_product < 0:
                interval = interval / 2.
                continue
            else:
                alpha = alpha + interval
        _, self.final_xs, self.final_ys = self.addFrameAccordingAlpha(alpha)
        return self.final_xs, self.final_ys

    def addFrameAccordingAlpha(self, alpha):
        if sum(self.normalized_xs * self.normalized_ys) > 0:
            lowerframe_xs, upperframe_xs, leftframe_ys, righframe_ys = self.calculateFrameCoordinatesPositive(alpha)
        else:
            lowerframe_xs, upperframe_xs, leftframe_ys, righframe_ys = self.calculateFrameCoordinatesNegative(alpha)

        new_xs = np.append(self.normalized_xs,
                           [lowerframe_xs, upperframe_xs,
                            self.frame_x_min * np.ones(self.num_of_points),
                            self.frame_x_max * np.ones(self.num_of_points)])
        new_ys = np.append(self.normalized_ys,
                           [self.frame_y_min * np.ones(self.num_of_points),
                            self.frame_y_max * np.ones(self.num_of_points),
                            leftframe_ys, righframe_ys])

        return sum(new_xs * new_ys), new_xs, new_ys

    def calculateFrameCoordinatesPositive(self, alpha):
        lowerframe_xs = self.frame_x_max - (self.frame_x_max - self.frame_x_min) * self.intervals ** alpha
        upperframe_xs = self.frame_x_max - (self.frame_x_max - self.frame_x_min) * (1 - self.intervals ** alpha)
        leftframe_ys = self.frame_y_max - (self.frame_y_max - self.frame_y_min) * self.intervals ** alpha
        righframe_ys = self.frame_y_max - (self.frame_y_max - self.frame_y_min) * (1-self.intervals ** alpha)
        return lowerframe_xs, upperframe_xs, leftframe_ys, righframe_ys

    def calculateFrameCoordinatesNegative(self, alpha):
        lowerframe_xs = self.frame_x_min + (self.frame_x_max - self.frame_x_min) * self.intervals ** alpha
        upperframe_xs = self.frame_x_min + (self.frame_x_max - self.frame_x_min) * (1 - self.intervals ** alpha)
        leftframe_ys = self.frame_y_min + (self.frame_y_max - self.frame_y_min) * self.intervals ** alpha
        righframe_ys = self.frame_y_min + (self.frame_y_max - self.frame_y_min) * (1-self.intervals ** alpha)
        return lowerframe_xs, upperframe_xs, leftframe_ys, righframe_ys

if __name__ == "__main__":
    #im = Image.open('./ADS.jpg')
    #ip = ImageProcessor(im)
    #ip.extractCoordiantes()
    #ip.normalizeCoordinates()
    #ip.addFrame()
    #print sum(ip.normalized_xs * ip.normalized_ys)
    #print sum(ip.final_xs * ip.final_ys)
    #plt.scatter(ip.final_xs, ip.final_ys, s=0.2)
    #plt.show()

    im = Image.open("./fa2017ads.jpg")
    ip = ImageProcessor(im)
    ip.addFrame()
    print sum(ip.normalized_xs * ip.normalized_ys)
    print sum(ip.final_xs * ip.final_ys)
    plt.scatter(ip.final_xs, ip.final_ys, s=0.2)
    plt.show()
