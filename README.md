# Predicting Wine Quality

![row of wineglasses](images/row_wine_glasses_utdallas.jpg?raw=true)
###### (image credit www.utdallas.edu)



## Overview

In this project, I analyze a publically available [dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/), which consists of various chemical measurements of different wines, as well as a quality score for each wine. The challenge is to try to predict the quality score, given these measurements. I first visualize several aspects of the dataset, focusing in particular on the problem of trying to satisfactorily display the relationship between a large number of continuous variables (the measurements) and a discrete response (the quality score). Then, I develop a model for predicting the quality score.  As an exercise, I do so using only the linear model (lm) as a base model, and try to make this simple model perform as well as possible, by implementing automatic feature selection and ensembling. 

## Exploratory Data Analysis

To begin with, I would like to visualize the relationship between the predictor variables (various chemical measurements such as pH) and the response (wine quality). I start by plotting a histogram of wine quality: 

![response_hist](images/response_hist_with_curve_cropped.png?raw=true)

Note that wine quality takes integer values, so the discretization in the histogram is present in the data, it is not caused by binning. The blue curve is a normal distribution fit to these data, and it is clearly a good fit. Therefore, wine quality appears to be approximately a discretization of a normally distributed random variable, which is reassuring, because it suggests that the assumptions of the linear model are valid.

To visualize the relationship between predictors and wine quality, I first try to use boxplots of the predictors, grouped by different quality values. This takes advantage of the discretization already present in the quality values.

![boxplot](images/boxplot_grouped_by_response.png?raw=true)

The disadvantage of this particular plot is that the numbers of data points with different quality values vary (the numbers can be read directly from the quality histogram). The boxplots at the extreme ends of the quality range are based on fewer data points and hence less reliable. 
To address these issues, I have tried an alternative approach, which is to draw scatter plots of quality vs. predictors, and use ggplot's geom_count() to visualize overlapping points as solid area.

![separate_regressions](images/separate_regressions.png?raw=true)

This plot clearly shows the smaller number of data points available for extreme values of quality. From either of these plots, we can see that there is a strong correlation between alcohol and quality, and moderate to weak correlations for some of the other predictors.

Next, I want to check for correlations between the predictor variables. The first step is a pairs plot:

![pairs plot](images/pairs_plot.png?raw=true)


Most variables seem to be weakly correlated. One aspect that does stand out is the correlation between free and total SO2 (it seems reasonable that these would be correlated). We can check the correlations more precisely using a correlation heatmap:

![corr_heatmap](images/corr_heatmap.png?raw=true)

Finally, print the pairs of variables with absolute correlation > 0.5:

      Var1      Var2       corr
      red       chlor      0.51
      res.sug   dens       0.55
      vol.acd   red        0.65
      alc       dens      -0.69
      red       tot.SO2   -0.70
      free.SO2  tot.SO2    0.72

In summary, a small subset of the variables have moderate correlations. This is something to keep in mind, but we don't expect correlations of this level to strongly affect classification.

## Classification

The variable to be predicted is quality, which is numeric and integer-valued. As a challenge, I developed a model that is based only on linear regression models (R lm function). The motivation for this is that I want this project to provide a useful case study for automatic feature selection and ensemble methods, and it's easier to see the effect of these approaches if they are used with a simple base classifier such as lm.
My approach for automatic feature selection is that I implement polynomial regression, i.e. the models are allowed to use features that consist of the variables raised to various powers, and automatically select the powers of the different variables using crossvalidation.

### The classification method: ensembles of polynomial regression models

I implemented a framework that allows entire ensembles of polynomial regression models to be trained and tested. Essentially, this framework consists of 2 rounds of nested crossvalidation. The outer crossvalidation is for evaluating the performance of the ensemble, whereas the inner crossvalidation is for searching for single models to include in the ensemble. The search for single models proceeds by first initializing a linear model to the baseline state, which typically will be that all variables are present, raised to the first power, and no interactions are present. Following initialization, the model is randomly changed by raising or lowering the power of variables (setting its power to zero drops a variable from the model) and adding or removing interactions. The effect of these changes is assessed using the hold-out set of the inner crossvalidation, and the change is kept if it improves performance. The search procedure is repeated several times to yield an ensemble of models, and the performance of the entire ensemble is assessed using the hold-out set of the outer crossvalidation. 

This diagram shows the nested crossvalidation approach:


<img src="images/ensemble_cv_schematic.png?raw=true" alt="ensemble cv schematic" width="500" height="600">

Written in pseudocode, the framework reads as:

    Split data into cross-validation folds (e.g. 5 folds)
        Set one fold aside as ensemble hold-out, rest are ensemble development data
        Initialize ensemble as empty list
        Until ensemble has desired number of models:
            Initialize a new model to baseline state
            For each step of model improvement:
                If the model has not been evaluated before: leave it unchanged
                Else: Make a random change to the model
                Split ensemble development data into cross-validation folds (e.g. 10 folds)
                For each training-holdout split: Fit model to training data, evaluate on holdout
            
                If average performance on model hold-out data is better than previous version of model: keep the change
            Add model to ensemble
    Fit each model in ensemble to entire development data
    Compute prediction of each model in ensemble on ensemble hold-out
    Take mean of these predictions -> this is ensemble prediction on hold-out
   
   
When testing different ensemble hyperparameters, I always compare the results to those obtained by fitting a single linear model at the baseline state. The ensemble has two advantages over the baseline model: the ensemble models are (hopefully) somewhat improved over the baseline model, and the ensembling itself (averaging over multiple different models) should also improve performance.

### Classification results

First of all, I wanted to determine how much better the single ensemble models are than the baseline model. To do so, I generated an ensemble consisting of just a single model and compared its performance with the baseline model. I measured performance with 5-fold crossvalidation at the outer level and repeated the whole process 10 times, so a total of 50 different single model / baseline model pairs were generated and evaluated. As an error metric, I used mean absolute deviation of the predicted quality values from the true values. I took means over the results from different models within both groups:

      [1] "mean error of baseline model   =  0.570006915271802"
      [1] "mean error of ensemble         =  0.561367379094108"
      [1] "SEM of error of baseline model =  0.00356751251098776"
      [1] "SEM of error of ensemble       =  0.0014630142723505"

So the iterative model refinement definitely helps performance, although the effect is only slight - 0.561/0.57 is approximately a 2% reduction in error. However, using a single improved model has essentially the same runtime and ease of interpretation as running the baseline model, so it is still worthwhile to implement.

Next, I wanted to determine how much performance is improved by averaging over a full ensemble as opposed to using a single model. To do so, I trained ensembles and tracked both the performance of the entire ensemble as well as the performance of each individual model. Here are representative results from one ensemble:

      [1] "error of ensemble                 =  0.557209221381607"
      [1] "errors of single models           =
      [1] 0.5627614 0.5663298 0.5611375 0.5653500 0.5611254 0.5617978 0.5622573
      [8] 0.5616120 0.5580755 0.5605172
      [1] "mean difference ensemble - single =  -0.00488716193646453"

So the population average of the ensemble does perform better than any individual model within it, but the effect is very small. In this case, it is a reduction in error of 0.005, corresponding to approximately 0.9% reduction in error.

### Conclusion

Using iterative model refinement results in small improvements in performance (approx 2% in testing). Using an ensemble of refined models results in a smaller improvement in performance (approx 0.9%). If this were a real-world application, it might be the case that using an ensemble would not be worth the extra complexity involved.
