# Wine Quality

![row of wineglasses](images/row_wine_glasses_utdallas.jpg?raw=true)
###### (image credit www.utdallas.edu)



## Overview

In this project, I analyze a publically available [dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/) to try to determine which factors influence the quality of wine. As a challenge, I try to do so using only the linear model (lm) as a base classifier, and use various feature generation and ensembling strategies to improve its performance.

## Exploratory Data Analysis

Visualize relationships between variables using pairs plot and correlation matrix:

![pairs plot](images/pairs_plot.png?raw=true)

Check correlations more precisely using correlation heatmap:

![corr_heatmap](images/corr_heatmap.png?raw=true)

## Classification

The variable we want to predict is quality, which is numeric and integer-valued.
