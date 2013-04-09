import LearnerInterface, SdmInterface
import cnidaria as cn
from collections import Counter
from functools import partial
from Aggregator import *

# This class is written for sdm's but it should be abstracted for 
# a more general class of learners.

def unlist(l):
  return([item for sublist in l for item in sublist])

class DistributedEnsembleLearner:

  def __init__(self, coordinator, learner_interface, 
    prediction_aggregator=PassThrough):

    self.aggregator=prediction_aggregator()

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
    tr = unlist(map(lambda c, ts: c.publish_exec(ts), self.coordinator, 
      train_strings))
    return(tr)

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
      worker_ids = [pr[i][0] for i in xrange(len(pr))]
      preds = [pr[i][1] for i in xrange(len(pr))]
      return self.aggregator(preds, worker_ids)
    else:
      return pr
