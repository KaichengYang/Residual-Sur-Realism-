# -*- coding: utf8 -*-
import numpy as np
from sklearn import linear_model
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


if __name__ == "__main__":
    gaussian_predicted, gaussian_residuals = calculateResidualsFromFile('./cfgauss_Lin_4p_5_flat.txt')
    plt.scatter(gaussian_predicted, gaussian_residuals, s=0.5)
    plt.show()
