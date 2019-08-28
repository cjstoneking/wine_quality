# Wine Quality

![row of wineglasses](images/row_wine_glasses_utdallas.jpg?raw=true)
###### (image credit www.utdallas.edu)



## Overview

In this project, I analyze a publically available [dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/) to try to determine which factors influence the quality of wine. As a challenge, I try to do so using only the linear model (lm) as a base classifier, and use various feature generation and ensembling strategies to improve its performance.

## Exploratory Data Analysis

Visualize relationships between variables using pairs plot and correlation matrix:

![pairs plot](images/pairs_plot.png?raw=true)

Most variables seem to be weakly correlated. One aspect that does stand out is the correlation between free and total SO2 (which seems reasonable). We can check the correlations more precisely using a correlation heatmap:

![corr_heatmap](images/corr_heatmap.png?raw=true)

Finally, print the pairs of variables with absolute correlation > 0.5:

      Var1      Var2       corr
      red       chlor      0.51
      res.sug   dens       0.55
      vol.acd   red        0.65
      alc       dens      -0.69
      red       tot.SO2   -0.70
      free.SO2  tot.SO2    0.72

In summary, a small subset of the variables have moderate correlations. THis is something to keep in mind, but we don't expect correlations of this level to strongly affect classification.

Next, visualize the relationship between the predictor variables and the response (wine quality). Start by plotting a histogram of wine quality:

![response_hist](images/response_hist.png?raw=true)

Note that wine quality takes integer values, so the discretization in the histogram is present in the data, it is not caused by binning. Wine quality appears to be approximately a discretization of a normally distributed random variable, which is reassuring, because it suggests that the assumptions of the linear model are valid.


## Classification

The variable we want to predict is quality, which is numeric and integer-valued.
