# -*- coding: utf8 -*-
import residuals
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ################## ################## ##################
    # Exmpale 1: Gauss portrait
    # use residuals.calculateResidualsFromFile to apply linear regression to the
    # data set "./gauss_data.txt" and draw the residuals against predicted
    # values
    # in the data set, the first column is the target and the rest are the
    # features
    gauss_predicted, gauss_residuals = residuals.calculateResidualsFromFile('./gauss_data.txt')
    plt.scatter(gauss_predicted, gauss_residuals, s=0.2)
    plt.show()
