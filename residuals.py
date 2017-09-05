# -*- coding: utf8 -*-
import numpy as np
from sklearn import linear_model
from PIL import Image
import matplotlib.pyplot as plt


def calculateResiduals(features, target):
    lmodel = linear_model.LinearRegression()
    lmodel.fit(features, target)
    predicted = lmodel.predict(features)
    return predicted, target - predicted


def calculateResidualsFromFile(file_name):
    raw_data = np.loadtxt(file_name)
    target = raw_data[:, 0]
    features = raw_data[:, 1:]
    return calculateResiduals(features, target)


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

        xmax = max(abs(max(self.normalized_xs)), abs(min(self.normalized_xs)))
        ymax = max(abs(max(self.normalized_ys)), abs(min(self.normalized_ys)))
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


class DataGenerator(object):
    def __init__(self, predicted, residuals, r0, n, p, jstar, beta0):
        self.predicted = predicted
        self.residuals = residuals
        self.r0 = r0
        self.n = n
        self.p = p
        self.jstar = jstar
        self.beta0 = beta0

        self.beta = np.matrix(np.arange(1, self.p+1)).T

        self.Y0 = np.matrix((self.residuals.std() / self.predicted.std())
                            * np.sqrt(self.r0**2 / (1 - self.r0**2))
                            * self.predicted).T
        self.R0 = np.matrix(residuals).T

        self.Z = np.matrix(np.random.normal(0, self.residuals.std(), self.n)).T
        self.M0 = np.matrix(np.random.normal(0, self.predicted.std(),
                                             (self.n, self.p)))
        self.Pr0 = self.R0 * self.R0.T / (self.R0.T * self.R0)

        self.X = None
        self.Y = None

    def calculateAM(self, M):
        W = np.c_[np.matrix(np.ones(self.n)).T,
                  (np.matrix(np.identity(self.n)) - self.Pr0) * M]
        Am = W * ((W.T * W).I) * W.T
        return Am

    def iterateM(self, M):
        M1 = M.copy()
        Am = self.calculateAM(M)

        sumbase = np.matrix(np.zeros(self.n)).T
        for j in range(1, self.p+1):
            if j != self.jstar:
                sumbase = sumbase + self.beta[j-1][0, 0] * M[:, j-1]

        mj = (self.Y0 - np.matrix(np.ones(self.n)).T * self.beta0
              - Am * self.Z + self.Pr0 * M * self.beta - sumbase) / float(self.beta[self.jstar-1])
        M1[:, self.jstar-1] = mj
        return M1

    def iterate(self, default_rounds=20):
        M = self.M0.copy()
        M1 = self.iterateM(self.M0)
        for i in range(default_rounds):
            M = M1
            M1 = self.iterateM(M)
            delta = max(((np.matrix(np.identity(self.n)) - self.Pr0) * (M1 - M)).A1)
            print "Round %d, delta: %.2E" % (i, delta)
            if delta < 1e-12:
                break
        return M1

    def generateData(self):
        M = self.iterate()
        X = (np.matrix(np.identity(self.n)) - self.Pr0) * M
        epsilon = self.R0 + self.calculateAM(M) * self.Z
        Y = np.matrix(np.ones(self.n)).T * self.beta0 + X * self.beta + epsilon
        self.X = X.A
        self.Y = Y.A
        return self.X, self.Y

    def dumpData(self, file_name):
        np.savetxt("./%s" % file_name, np.c_[self.Y, self.X])


if __name__ == "__main__":
    pass
    # gaussian example
    #gaussian_predicted_data, gaussian_residuals = dataloader.calculateResidualsFromFile("./cfgauss_Lin_4p_5_flat.txt")
    #print gaussian_residuals.shape[0]
    #dg = DataGenerator(gaussian_predicted_data, gaussian_residuals, 0.75, gaussian_residuals.shape[0], 4, 4, 0)

    # ads demo
    #ads_prediceted_data, ads_residuals = dataloader.calculateResidualsFromFile("./ads_demo_data.txt")
    #plt.scatter(ads_prediceted_data, ads_residuals)
    #plt.show()

    #X, Y = dg.generateData()
    #y, r = dataloader.calculateResiduals(X, Y)
    #plt.scatter(y, r)
    #plt.show()
