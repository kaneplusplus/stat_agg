from pandas import DataFrame
from functools import partial
from collections import Counter
from numpy import *
from sklearn.ensemble import *
from sklearn import linear_model
from statistics import pstdev, variance

class PredictionAggregator:
  """The abstract class for aggregating predictions."""
  def train(self, training_data):
    pass
  def predict(self, prediction_data):
    pass

def tally(votes, majority=True):
  """Tally votes and return either the majority, the minority vote, or None
  if there are ties.

  Args: 
    votes (list) : the votes to tally.
    majority (bool): should the most frequent vote be returned? If False 
      the least frequent is used

  Returns:
    If majority is True then the most frequent vote is returned. If 
    False then the least frequent. If there is no majority/minority 
    then None.

  Example:
    >>> tally(["a", "a", "b"])
    'a'    
    >>> tally(["a", "a", "b"], majority=False)
    'b'
    >>> tally(["a", "a", "b", "b"]) == None
    True
  """  
  C = Counter(votes)
  l = C.most_common()
  if not majority:
    l.reverse()
  if len(l) == 1:
    return(l[0][0])
  if l[0][1] == l[1][1]:
    return(None)
  return(l[0][0])

class SimpleVote(PredictionAggregator):
  """The class that implements the MajorityVote and MinorityVote 
  classes for predicting categorical variables."""
  def __init__(self, agg):
    self.agg = agg
  
  def train(training_data):
    pass

  def predict(self, prediction_data):
    df = DataFrame(prediction_data)
    ret = []
    for row in df.iterrows():
      index, data = row
      ret += [self.agg(data.tolist())]
    return(ret)

class MajorityVote(SimpleVote):
  """Aggregate categorical variables and predict based on a majority vote

  Example:
    >> # Create 2 learners named '1' and '2' with 2 predictions each.
    >> prediction_data = {'1': ['a', 'a'], '3': ['b', 'a']}
    >> mv = MajorityVote()
    >> print(mv.predict(prediction_data))
    [None, 'a']
  """
  def __init__(self):
    SimpleVote.__init__(self, agg=partial(tally, majority=True))

class MinorityVote(SimpleVote):
  """Aggregate categorical variables and predict based on a minority vote

  Example:
    >> # Create 2 learners named '1' and '2' with 2 predictions each.
    >> prediction_data = {'1': ['a', 'a'], '2' : ['a', 'a'], '3': ['b', 'a']}
    >> mv = MinorityVote()
    >> print(mv.predict(prediction_data))
    ['b', 'a']
  """
  
  def __init__(self):
    SimpleVote.__init__(self, agg=partial(tally, majority=False))

class ContinuousAverage(PredictionAggregator):
  """Aggregate continuous variables and predict on the average of learners'
  predition

  Example:
    >> # Create 2 learners named '1' and '2' with 2 predictions each.
    >> prediction_data = {'1': [10, 10], '2' : [20, 30], '3': [30, 50]}
    >> ca = ContinuousAverage()
    >> print(ca.predict(prediction_data))
    [20.0, 30.0]
  """
  def predict(self, prediction_data):
    preds = DataFrame(prediction_data)
    ret = []
    for row in preds.iterrows():
      index, data = row
      ret.append(mean(data))
    return(ret)

class MinimumContinuousOLS(PredictionAggregator):
  """Aggregate continuous variables and predict base on the ols estimator

  Example:
    >>> from pandas import read_csv
    >>> from statistics import pstdev, variance
    >>> import statsmodels.formula.api as sm
    >>> 
    >>> iris_url = "https://raw.githubusercontent.com/pydata/pandas/master/pandas/tests/data/iris.csv" 
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
  """
  def __init__(self):
    self.df = None

  def train(self, training_data):
    self.df = DataFrame(data=training_data['prediction'])
    self.actual = training_data['actual']
  def predict(self, prediction_data):
    lm = linear_model.LinearRegression()
    lm.fit(self.df, self.actual)
    pred_df = DataFrame(data=prediction_data)
    return(lm.predict(pred_df))

class MinimumContinuousRandomForest(PredictionAggregator):
  """Aggregate continuous variables and predict base on the estimator variance 

  Example:
    >>> from pandas import read_csv
    >>> from statistics import pstdev, variance
    >>> import statsmodels.formula.api as sm
    >>> 
    >>> iris_url = "https://raw.githubusercontent.com/pydata/pandas/master/pandas/tests/data/iris.csv"
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
    0.356900395729
    >>> print pstdev(est2 - iris_sample['SepalLength'])
    0.311543723503
    >>> print pstdev(est3 - iris_sample['SepalLength'])
    0.389496162197
    >>> 
    >>> 
    >>> training_data = {"prediction" : {'a': est1,
    ...                                  'b': est2,
    ...                                  'c': est3},
    ...                  "actual" : iris_sample['SepalLength']}
    >>> 
    >>> # Use the training data to fit the minimum variance aggregator.
    ... 
    >>> mcrf = MinimumContinuousRandomForest()
    >>> mcrf.train(training_data)
    >>> print pstdev(mcrf.predict(training_data['prediction']) - \
    ...                           iris_sample['SepalLength'])
    0.325353841932
  """
  def __init__(self):
    self.df = None

  def train(self, training_data):
    self.df = DataFrame(data=training_data['prediction'])
    #self.df['actual'] = [x for x in training_data['actual']]
    self.actual = training_data['actual']

  def predict(self, prediction_data):
    clf = RandomForestRegressor()
    clf.fit(self.df[prediction_data.keys()], self.actual)
    return(clf.predict(DataFrame(prediction_data)))

class MinimumContinuousVariance(PredictionAggregator):
  """Aggregate continuous variables and predict base on the estimator variance 

  Example:
    >>> from pandas import read_csv
    >>> from statistics import pstdev, variance
    >>> import statsmodels.formula.api as sm
    >>> 
    >>> iris_url = "https://raw.githubusercontent.com/pydata/pandas/master/pandas/tests/data/iris.csv"
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
    0.356900395729
    >>> print pstdev(est2 - iris_sample['SepalLength'])
    0.311543723503
    >>> print pstdev(est3 - iris_sample['SepalLength'])
    0.389496162197
    >>> 
    >>> 
    >>> training_data = {"prediction" : {'a': est1,
    ...                                  'b': est2,
    ...                                  'c': est3},
    ...                  "actual" : iris_sample['SepalLength']}
    >>> 
    >>> # Use the training data to fit the minimum variance aggregator.
    ... 
    >>> mcv = MinimumContinuousVariance()
    >>> mcv.train(training_data)
    >>> print pstdev(mcv.predict(training_data['prediction']) - \
    ...                          iris_sample['SepalLength'])
    0.316002742912
  """
  def __init__(self):
    self.variances = {}
  def train(self, training_data):
    preds = training_data['prediction']
    for k in preds:
      self.variances[k] = variance(preds[k] - training_data['actual'])
  def predict(self, prediction_data):
    if not len(self.variances):
      raise(RuntimeError("Model not trained."))
    denom = sum([1/self.variances[x] for x in prediction_data.keys()])
    for k in prediction_data:
      prediction_data[k]=[(x/self.variances[k])/denom for x in \
                          prediction_data[k]]
    ks = prediction_data.keys()
    ret = array(prediction_data[ks.pop()])
    for k in ks:
      ret += prediction_data[k]
    return(ret)
    
class MinimumClassificationVariance:
  """Aggregate categorical variables and predict base on the estimator variance 

  Example
    >>> training_data = {"prediction":{'1': ['a', 'b', 'a'], '3': ['a', 'a', 'b'],
    ...   '2' : ['a', 'a', 'a']}, "actual" : ['a', 'a', 'a']}
    >>> 
    >>> prediction_data = {'1': ['a', 'a'], '3': ['b', 'a']}
    >>> 
    >>> mcv = MinimumClassificationVariance()
    >>> mcv.train(training_data)
    >>> print(mcv.predict(prediction_data))
    [None, 'a']
  """

  def __init__(self, precision=100):
    self.weights = {}

  def train(self, training_data):
    preds = DataFrame(training_data['prediction'])
    preds['actual'] = training_data['actual']
    pred_cols = len(training_data['prediction'].keys())
    results = DataFrame()
    for row in preds.iterrows():
     index, data = row 
     results = results.append( data[range(pred_cols)] == data['actual'] )
    for k in preds[range(pred_cols)]:
      self.weights[k] = 1/variance(1-results[k])
    # If we have infinte weights make them 2* the sum of the other 
    if any(x == inf for x in self.weights):
      tot_weight = sum( [x for x in self.weights.values() if x != inf] )
      for wk in self.weights:
        if self.weights[wk] == inf:
          self.weights[wk] = 2*tot_weight
    
  def predict(self, prediction_data):
    preds = DataFrame(prediction_data)
    col_names = prediction_data.keys()
    tally_dict = {}
    for col_name in unique(preds):
      tally_dict[col_name] = [0 for x in range(preds.shape[0])]
    for row in preds.iterrows():
      index, data = row
      for col_name, elem in zip(col_names, data):
        tally_dict[elem][index] += self.weights[col_name]
    tally_df = DataFrame(tally_dict)
    max_val = [int(round(x)) for x in tally_df.max(1).tolist()]
    max_level = []
    for row in tally_df.index:
      int_vals = [int(round(x)) for x in tally_df.ix[row].tolist()] 
      is_max = [x == max_val[row] for x in int_vals]
      if sum(is_max) > 1:
        max_level.append(None)
      else:
        max_level.append(tally_df.columns[ is_max ][0])
    return(max_level)

