from numpy import *
from functools import partial
from collections import Counter

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


class MinimumVariance:
  pass

class MinimumErrorRate:
  pass

#class MinimumL2:
