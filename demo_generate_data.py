# -*- coding: utf8 -*-
from PIL import Image
import residuals
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ################## ################## ##################
    # Example 1: generate data set with "./ADS.jpg"
    # open the image with PIL
    ADS_image = Image.open('./ADS.jpg')
    # initialize the ImageProcessor instance
    ADS_image_processor = residuals.ImageProcessor(ADS_image)
    # add frame
    ADS_image_processor.addFrame()
    # get the predicted and residuals
    ADS_predicted, ADS_residuals = ADS_image_processor.final_xs, ADS_image_processor.final_ys
    # feed predicted and residuals to DataGenerator
    ADS_dat_generator = residuals.DataGenerator(ADS_predicted, ADS_residuals, 0.75, ADS_predicted.shape[0], 4, 4, 0)
    # get the features and target for the generated data
    ADS_X, ADS_Y = ADS_dat_generator.generateData()
    # to test, apply linear regression again
    ADS_predicted_test, ADS_residuals_test = residuals.calculateResiduals(ADS_X, ADS_Y)
    # plot the residuals against predicted from the generated data shold see the
    # original picture
    plt.scatter(ADS_predicted_test, ADS_residuals, s=0.2)
    plt.show()

    ################## ################## ##################
    # Example 2: generate data set with "./FA2017ADS.jpg"
    # open the image with PIL
    FA2017ADS_image = Image.open('./FA2017ADS.jpg')
    # initialize the ImageProcessor instance
    FA2017ADS_image_processor = residuals.ImageProcessor(FA2017ADS_image)
    # add frame
    FA2017ADS_image_processor.addFrame()
    # get the predicted and residuals
    FA2017ADS_predicted, FA2017ADS_residuals = FA2017ADS_image_processor.final_xs, FA2017ADS_image_processor.final_ys
    # feed predicted and residuals to DataGenerator
    FA2017ADS_dat_generator = residuals.DataGenerator(FA2017ADS_predicted, FA2017ADS_residuals, 0.75, FA2017ADS_predicted.shape[0], 4, 4, 0)
    # generate the features and target
    FA2017ADS_dat_generator.generateData()
    # save the data into file
    # the first column is the target and the rest are features
    FA2017ADS_dat_generator.dumpData("FA2017GeneratedData.txt")
    # to test, apply linear regression again
    FA2017ADS_predicted_test, FA2017ADS_residuals_test = residuals.calculateResidualsFromFile("./FA2017GeneratedData.txt")
    # plot the residuals against predicted from the generated data shold see the
    # original picture
    plt.scatter(FA2017ADS_predicted_test, FA2017ADS_residuals, s=0.2)
    plt.show()
