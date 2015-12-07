stat\_agg
===

Statistical aggregates for machine learning in Python.

Description
---

stat\_agg is a simple Python package for aggregating predictions from an ensemble
of learners. When aggregated, the statistical accuracy of predictions
from different learners is often greater than any one of them. This package implements
various approaches to aggregated prediction of continuous and categorical 
prediction challenges.


The goal of the stat\_agg package is to:

1. Provide a suite of statistical aggregators that maximize ensemble
prediction accuracy for continuous and categorical outcomes.
2. Manage ensemble in a way that is dynamic. New learners
can be added to an ensemble at any time.
3. Detect and retrain when one or more of the learners suddenly becomes
unavailable.

Requirements
---

The stat\_agg package has been tested on Python version 2.7 with the following
packages:
- pandas 0.17.1
- sklearn 0.17

Installing stat\_agg
---

The easiest way to install stat\_agg is to use pip from within a shell:

```bash
> pip install -e git+https://github.com/kaneplusplus/stat_agg.git#egg=stat_agg
```

Support
---

stat_agg is supported on Python version 2.7

The development home of this project can be found at: [https://github.com/kaneplusplus/stat\_agg](https://github.com/kaneplusplus/stat\_agg)

The package currently supports the following statistical aggregators:
- Categorical Prediction 
    - Max Vote
    - Min Vote
    - Minimum Variance
- Continuous Prediction
    - Average Value
    - Minimum Variance
    - Least Squares
    - Random Forests

Using the library
---

### Simple Aggregators

Simple aggregators are those where no training is necessary and the aggregate
prediction can be calculated directly from learners. One example of this is
majority vote where the aggregator simply returns the predition that appears the
most often. An example is shown below.

```{Python}
>> # Create 2 learners named '1' and '2' with 2 predictions each.
>> prediction_data = {'1': ['a', 'a'], '3': ['b', 'a']}
>> mv = MajorityVote()
>> print(mv.predict(prediction_data))
[None, 'a']
```
Note that in the first prediction, learner 1 and 2 predicted ```'a'``` and 
```'b'``` respectively. Since there is no majority in this case, a value of
```None``` was returned. Other simple aggregators include minority vote for
categorical variables and average for continuous outcomes.

### Model-based Aggregators

More sophisticated aggregators can be constructed by training on the
accuracy of learners' predictions. One example is the ordinary least squares (OLS)
aggregator, which uses learner predictions as regressors and fits against the
outcome. An example using the iris data set is shown below.

```{Python}
>>> fromm pandas import read_csv
>>> from statistics import pstdev, variance
>>> import statsmodels.formula.api as sm
>>> 
>>> iris_url = "https://raw.githubusercontent.com/pydata/pandas/master/panda s/tests/data/iris.csv" 
>>> 
>>> # Download the iris data set.
... iris = read_csv(iris_url)
>>> 
>>> # Partition iris into 3 parts.
... iris1 = iris[0:15].append(iris[50:65]).append(iris[100:115])
>>> iris2 = iris[15:40].append(iris[65:90]).append(iris[115:140])
>>> iris3 = iris[40:50].append(iris[90:100]).append(iris[140:150])
>>> 
>>> # Fit the iris subsets using the statsmodels package..
... form = "SepalLength ~ SepalWidth + PetalLength + PetalWidth + Name"
>>> fit1 = sm.ols(formula=form, data=iris1).fit()
>>> fit2 = sm.ols(formula=form, data=iris2).fit()
>>> fit3 = sm.ols(formula=form, data=iris3).fit()
>>> 
>>> # Get a random subset of the iris data.
... iris_sample = iris.sample(50)
>>> 
>>> est1 = fit1.predict(iris_sample)
>>> est2 = fit2.predict(iris_sample)
>>> est3 = fit3.predict(iris_sample)
>>> 
>>> # Print the in-sample standard errors.
... print pstdev(est1 - iris_sample['SepalLength'])
0.3014955119
>>> print pstdev(est2 - iris_sample['SepalLength'])
0.279841460366
>>> print pstdev(est3 - iris_sample['SepalLength'])
0.363993665693
>>> 
>>> 
>>> training_data = {"prediction" : {'a': est1,
...                                  'b': est2,
...                                  'c': est3},
...                  "actual" : iris_sample['SepalLength']}
>>> 
>>> # Use the training data to fit the OLS aggregator.
... mco = MinimumContinuousOLS()
>>> mco.train(training_data)
>>> 
>>> # Print the standard deviation of the aggregator.
>>> print pstdev(mco.predict(training_data['prediction']) - \
...                          iris_sample['SepalLength'])
0.271979762123
```
The OLS aggregator provides a small increase of in-sample variance. Other model-based
aggregators include minimum variance (for both categorical and continuous outcomes)
and a random forests aggregator.

Contact
---

Contributions are welcome.

For more information contact Michael Kane at [kaneplusplus@gmail.com](kaneplusplus@gmail.com).

