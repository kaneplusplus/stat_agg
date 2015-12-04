from pandas import DataFrame
from functools import partial
from collections import Counter
from numpy import *
import statsmodels.formula.api as sm
import pdb
from sklearn.ensemble import *

# See if we can define the function parameters for at least predict.
class PredictionAggregator:
  def train(self, training_data):
    pass
  def predict(self, prediction_data):
    pass

# Support functions for majority and minority voting.
def tally(votes, majority=True):
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
  
  def __init__(self):
    SimpleVote.__init__(self, agg=partial(tally, majority=True))

class MinorityVote(SimpleVote):
  
  def __init__(self):
    SimpleVote.__init__(self, agg=partial(tally, majority=False))

class MinimumContinuousOLS(PredictionAggregator):
  def __init__(self):
    self.df = None

  def train(self, training_data):
    self.df = DataFrame(data=training_data['prediction'])
    self.df['actual'] = [x for x in training_data['actual']]
  def predict(self, prediction_data):
    form = "actual ~ " + "+".join(prediction_data.keys())
    fit = sm.ols(formula=form, data=self.df).fit()
    pred_df = DataFrame(data=prediction_data)
    return(fit.predict(pred_df))

class MinimumContinuousRandomForest(PredictionAggregator):
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
    # TODO: START HERE.
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


#if __name__=='__main__':
from pandas import read_csv
from statistics import pstdev, variance

iris = read_csv("https://raw.githubusercontent.com/pydata/pandas/master/pandas/tests/data/iris.csv")

iris1 = iris[0:15].append(iris[50:65]).append(iris[100:115])
iris2 = iris[15:40].append(iris[65:90]).append(iris[115:140])
iris3 = iris[40:50].append(iris[90:100]).append(iris[140:150])

# Fit the iris subsets.
form = "SepalLength ~ SepalWidth + PetalLength + PetalWidth + Name"
fit1 = sm.ols(formula=form, data=iris1).fit()
fit2 = sm.ols(formula=form, data=iris2).fit()
fit3 = sm.ols(formula=form, data=iris3).fit()

# Get a random subset of the iris data.
iris_sample = iris.sample(50)

est1 = fit1.predict(iris_sample)
est2 = fit2.predict(iris_sample)
est3 = fit3.predict(iris_sample)

print pstdev(est1 - iris_sample['SepalLength'])
print pstdev(est2 - iris_sample['SepalLength'])
print pstdev(est3 - iris_sample['SepalLength'])

# The fitted and actual values are the training data for the ensemble 
# learner.
training_data = {"prediction" : {'a': est1,
                                 'b': est2,
                                 'c': est3},
                 "actual" : iris_sample['SepalLength']}
mcv = MinimumContinuousVariance()
mcv.train(training_data)
print pstdev(mcv.predict(training_data['prediction']) - \
                         iris_sample['SepalLength'])

mco = MinimumContinuousOLS()
mco.train(training_data)
print pstdev(mco.predict(training_data['prediction']) - \
                         iris_sample['SepalLength'])

mcrf = MinimumContinuousRandomForest()
mcrf.train(training_data)
print pstdev(mcrf.predict(training_data['prediction']) - \
                          iris_sample['SepalLength'])

training_data = {"prediction":{'1': ['a', 'b', 'a'], '3': ['a', 'a', 'b'],
  '2' : ['a', 'a', 'a']}, "actual" : ['a', 'a', 'a']}

#prediction_data = [('1', ['a', 'a']), ('3', ['b', 'a'])]
prediction_data = {'1': ['a', 'a'], '3': ['b', 'a']}

mv = MinorityVote()
print(mv.predict(prediction_data))

mv = MajorityVote()
print(mv.predict(prediction_data))

mcv = MinimumClassificationVariance()
mcv.train(training_data)
print(mcv.predict(prediction_data))
