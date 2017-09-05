# -*- coding: utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
import dataloader


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
    # bulls eye example
    #bulls_eye_predicted_data, bulls_eye_residuals = dataloader.calculateResidualsFromFile("./bullseye_Lin_4p_5_flat.txt")
    #dg = DataGenerator(bulls_eye_predicted_data, bulls_eye_residuals, 0.75, 600, 4, 4, 0)

    # gaussian example
    gaussian_predicted_data, gaussian_residuals = dataloader.calculateResidualsFromFile("./cfgauss_Lin_4p_5_flat.txt")
    print gaussian_residuals.shape[0]
    #dg = DataGenerator(gaussian_predicted_data, gaussian_residuals, 0.75, gaussian_residuals.shape[0], 4, 4, 0)

    # ads demo
    #ads_prediceted_data, ads_residuals = dataloader.calculateResidualsFromFile("./ads_demo_data.txt")
    #plt.scatter(ads_prediceted_data, ads_residuals)
    #plt.show()

    #X, Y = dg.generateData()
    #y, r = dataloader.calculateResiduals(X, Y)
    #plt.scatter(y, r)
    #plt.show()
