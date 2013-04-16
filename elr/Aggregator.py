from numpy import *
from functools import partial
from collections import Counter
from numpy import *

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
      return("None")
    return(l[0][0])

# Clean this up.
def vote(preds, count_fn=partial(tally, majority=True)):
  if len(preds) == 0:
    raise(RuntimeError("No predictions supplied"))
  # TODO: remove the values that are exceptions 
  ret=[]
  for i in xrange(len(preds[0])):
    votes=[]
    for j in xrange(len(preds)):
      votes.append(preds[j][i])
    ret.append(count_fn(votes))
  return(ret)

class PassThrough(PredictionAggregator):
  
  def train(self, training_data):
    pass

  def predict(self, prediction_data):
    return prediction_data

class SimpleVote(PredictionAggregator):

  def __init__(self, agg):
    self.agg = agg
  
  def train(training_data):
    pass

  def predict(self, prediction_data):
    preds = [prediction_data[i][1] for i in xrange(len(prediction_data))]
    return self.agg(preds)

class MajorityVote(SimpleVote):
  
  def __init__(self):
    SimpleVote.__init__(self, partial(vote, 
      count_fn=partial(tally, majority=True)))
#    self.agg = partial(vote, count_fn=partial(tally, majority=True))

class MinorityVote(SimpleVote):
  
  def __init__(self):
    SimpleVote.__init__(self, partial(vote,
      count_fn=partial(tally, majority=False)))
#    self.agg = partial(vote, count_fn=partial(tally, majority=False))


class MinimumClassificationVariance:

  def __init__(self, precision=100):
    self.precision=precision

  # In this case prediction_data are tuples of size 2. The first
  # element is the learner id the second is the classification 
  # accuracy and the number of samples used to get the accuracy.
  def train(self, training_data):
    # Get the variances and take their inverses.

    ns = array([float(len(x[1])) for x in training_data['prediction']])
    actual = array(training_data['actual'])
    preds = [array(training_data['prediction'][i][1])
      for i in xrange(len(training_data['prediction']))]

    # If we have infinite weights make them 2 * the sum of the other 
    # weights.
    ps = [float(sum(actual == preds[i])) for i in xrange(len(preds))]/ns
    ps = array([x if x < 1. else 1. - 1./(float(self.precision)) for x in ps])
    weights = 1/(ns*ps*(1-ps))
    self.id_weight = dict([(training_data['prediction'][i][0],
      weights[i]) for i in xrange(len(weights))])
  

  def predict(self, prediction_data):

    pred_ids = [x[0] for x in prediction_data]
    intersect_ids = filter(lambda x : x in self.id_weight.keys(), pred_ids)

    normalization = sum([self.id_weight[int_id] for int_id in intersect_ids])

    # Get the prediction data in order of the intersection_ids
    prediction_data_dict = dict(prediction_data)
    ordered_prediction_data = [prediction_data_dict[identifier] for 
      identifier in intersect_ids]

    # Now get the number of repetitions per vote based on the weighting.
    ordered_rep=[int(round(
      self.id_weight[identifier]/normalization*self.precision)) for
      identifier in intersect_ids]

    # Now get the predictions.
    preds=[]
    for i in xrange(len(prediction_data[0][1])):
      votes=[]
      for j in xrange(len(ordered_rep)):
        votes += ([ordered_prediction_data[j][i]]*ordered_rep[j])
      preds.append(tally(votes))

    return(preds)


class MinimumErrorRate:
  pass

#class MinimumL2:

if __name__=='__main__':

  training_data = {"prediction":[('1', ['a', 'b', 'a']), ('3', ['a', 'a', 'b']),
    ('2', ['a', 'a', 'a'])], "actual" :['a', 'a', 'a']}

  prediction_data = [('1', ['a', 'a']), ('3', ['b', 'a'])]
  mcv = MinimumClassificationVariance()
  mcv.train(training_data)
  print(mcv.predict(prediction_data))
