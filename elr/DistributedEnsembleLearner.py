import LearnerInterface, SdmInterface
import cnidaria as cn
from collections import Counter
from functools import partial

# This class is written for sdm's but it should be abstracted for 
# a more general class of learners.

def unlist(l):
  return([item for sublist in l for item in sublist])

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
def vote(preds, count_fn):
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


class DistributedEnsembleLearner:

  def __init__(self, coordinator, learner_interface, 
    prediction_aggregator="majority_vote"):

    if not prediction_aggregator in ["majority_vote", "minority_vote", "none"]:
      raise(RuntimeError("Unknown prediction aggregator"))
    else:
      if (prediction_aggregator == "majority_vote"):
        self.aggregator=partial(vote, count_fn=partial(tally, majority=True))
      elif (prediction_aggregator == "minority_vote"):
        self.aggregator=partial(vote, count_fn=partial(tally, majority=False))
      elif (prediction_aggregator == "none"):
        self.aggregator = lambda x: x

    cislist = isinstance(coordinator, list)
    lwislist = isinstance(learner_interface, list)

    if cislist != lwislist:
      raise(RuntimeError("Either coordinator and learner interface are"+
        " lists or neither are."))

    if cislist:
      if len(coordinator) != len(learner_interface):
        raise(RuntimeError("Coordinator and learner interface lists must be "+
          "the same length"))
      self.coordinator = coordinator
      self.learner_interface = learner_interface
    else:
      self.coordinator = [coordinator]
      self.learner_interface= [learner_interface]

  def fit(self, data_handle):

    train_strings = [x.fit_string(data_handle) for x in self.learner_interface]
    tr = map( lambda c, ts: c.publish_exec(ts), self.coordinator, train_strings)
    return(unlist(tr))

  def predict(self, data_handle, aggregate=True):
  
    #predict_strings = self.learner_interface.predict_string(data_handle)
    predict_strings = [x.predict_string(data_handle) for x in 
      self.learner_interface]
    #get_pred_string = self.learner_interface.get_pred_string()
    get_pred_string = [x.get_pred_string() for x in self.learner_interface]
    prStatus = map( lambda c, ps: c.publish_exec(ps), self.coordinator, 
      predict_strings)
    pr = unlist(map( lambda c, ps: c.publish_get(ps), self.coordinator, 
      get_pred_string))
    if aggregate:
      return(self.aggregator(pr))
    else:
      return(pr)
