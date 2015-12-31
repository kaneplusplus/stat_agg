
# Benchmark

The stat_agg package was intented to be used for an ensemble of 
heterogeneous learners. It should be noted that this is slightly different
than the ensemble learners provided in packages like sklearn, where each
learner resamples a single data set and/or the parameters for each learner are
varied. stat_agg makes no assumptions about either the learners or
the data sets from which the predictions are calculated. It only assumes
that predictions are for the same outcome. Furthermore, we are not aware
of other packages of this kind. We speculate that statistical 
aggregation of heterogeneous learners is generally done ad-hoc and part of the
intention of this package is to leverage sklearn for both the creation
of the individual learners as well as their aggregate prediction. 

As a result,the benchmarks here show the performance for a simple use case;
namely, prediction airline delays. The data set used will be 2008 flight delay 
information for those carriers with at least 1% commercial U.S. domestic flights.
The data set contains information for 7,009,729 flights. Each flight contains 
29 variables related to flight time, delay time, departure airport, arrival 
airport, and so on.

## Reproducing the Results

All scripts required to run the benchmarks are available in the [stat_agg
repository](https://github.com/kaneplusplus/stat_agg/tree/master/benchmarks). The directory contains:

- [make_data.r](https://github.com/kaneplusplus/stat_agg/blob/master/benchmarks/make_data.r) - an R script for downloading the data, creating training and
test data, fitting the learners (for both regression and classification),
and outputting their predictions.
- [benchmark_regression.py](https://github.com/kaneplusplus/stat_agg/blob/master/benchmarks/benchmark_regression.py) - a Python script that runs the regression 
benchmarks.
- [benchmark_classification.py](https://github.com/kaneplusplus/stat_agg/blob/master/benchmarks/benchmark_classification.py) - a Python script that runs the 
classification benchmarks.
- [benchmark.rmd](https://github.com/kaneplusplus/stat_agg/blob/master/benchmarks/benchmark.rmd) - an R markdown file for generating this file.

To create the files needed for the benchmark you can simply either source 
make_data.r or call it from the terminal.

```bash
Rscript make_data.r
```

## The Benchmark Datasets

The following data sets are created when the make\_data.r script is run:

- regression_predictions_train.csv - the in-sample predictions of
the departure delay of 6 learners. All learners were fitted using a 
random subset of 5 variables and least squares regression. 
- regression_predictions_test.csv - out of sample predictions of the departure
delay and a extra column giving the actual departure delay.
- classification_prediction_train.csv - the in-sample predictions of whether
or not a flight was at left late (the departure delay is greater than or
equal to 15 minutes) of 6 learners. All learners were fitted using a
random subset of 5 variables and all were fitted using logistic regression.
- classification_prediction_test.csv - out of sample predictions of late
departures and an extra column giving the actual departure delay.

## Classification: Predicting Lateness

The benchmark_classification.py script benchmarks the aggregation of classification
predictions using minimum variance. The package is trained on a data set where
each of the learners is represented. However, if variables are not present
for the prediction, whether it is because the learner is not available or
because it is taking too long, the internal weighting of the learners is
recalculated. The timing can be reproduced by by running the script.


```bash
python benchmark_classification.py
```

```
## 367.94 seconds to predict 1 variables.
## 338.02 seconds to predict 2 variables.
## 349.28 seconds to predict 3 variables.
## 351.94 seconds to predict 4 variables.
## 355.63 seconds to predict 5 variables.
## 356.56 seconds to predict 6 variables.
## 366.35 seconds to predict 7 variables.
## 361.46 seconds to predict 8 variables.
## 356.29 seconds to predict 9 variables.
## 364.98 seconds to predict 10 variables.
```

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3-1.png) 

The plot above shows the misclassification rate of the aggregates
in the number of learners. The horizontal red line shows the
best classfication rate of any individual learner. In general 
the accuracy first increases but past a given point, more learners 
may actually decrease accuracy. As a result the number of learners must
be tuned for the application. However, when tuned they generally provide
accuracy well beyond the accuracy of any given predictor. 

## Regression: Predicting Departure Delays

The benchmark_regression.py script benchmarks the aggregation of continuous
predictions using least squares. The package is trained on a data set where
each of the learners is represented. However, if variables are not present
for the prediction, whether it is because the learner is not available or
because it is taking too long, the internal weighting of the learners is
recalculated. The timing for least squares can be tested simply by running 
the script.


```bash
python benchmark_regression.py
```

```
## 0.06 seconds for training.
## 0.20 seconds to predict 1 variables.
## 0.18 seconds to predict 2 variables.
## 0.25 seconds to predict 3 variables.
## 0.31 seconds to predict 4 variables.
## 0.38 seconds to predict 5 variables.
## 0.46 seconds to predict 6 variables.
## 0.53 seconds to predict 7 variables.
## 0.60 seconds to predict 8 variables.
## 0.75 seconds to predict 9 variables.
## 0.83 seconds to predict 10 variables.
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-1.png) 

The plot above shows the accuracy of the aggregate predictor in the number
of learners used to create the aggregate. The horizontal red line shows the
best accuracy of any individual learner. It is importatnt to aggregates are not guaranteed to improve accuracy. 
