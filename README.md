# A Python implementation for Residual Sur(Realism)

## Residual Sur(Realism) what?

The general goal here is to generate a set of data.
After applying linear regression to the data set, the scatter graph of the residuals against predicted values should be a picture.
For more detailed information, follow the [link](http://www4.stat.ncsu.edu/~stefanski/NSF_Supported/Hidden_Images/stat_res_plots.html#Download_Data_Sets) and read [the paper](http://www4.stat.ncsu.edu/~stefanski/NSF_Supported/Hidden_Images/Residual_Surrealism_TAS_2007.pdf). 

## About the code

There is an implementation in R on the web page mentioned just now.
This repo contains a Python 2.7 implementation.

The implementation here is based on the original paper.

All the useful methods and class are in `residuals.py`.
Also check the demonstrations to see how it works.

Dependencies:

* numpy
* sklearn
* matplotlib
* PIL
