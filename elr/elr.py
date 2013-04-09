from __future__ import division
from SdmInterface import SdmInterface
from DistributedEnsembleLearner import DistributedEnsembleLearner
from DistributedEnsembleLearner import vote
from DistributedEnsembleLearner import tally
from cnidaria import *
import redis, time, sdm
from sklearn import preprocessing
from functools import partial
from Aggregator import *

feature_training_file="../../feats_train.h5"
feature_testing_file="../../feats_test.h5"
# Get the classification from the file name.
def get_actual(filenames, features):
  ret=[]
  for fn in filenames:
    found=False
    for feat in features:
      if feat in fn:
        ret.append(feat)
        found=True
        break
    if not found:
      ret.append(None)
  return(ret)

# Start the workers.
start_local_workers(nw=5, path="/Users/mike/Projects/xdata/xdataenv/bin/",
  verbose=True)
time.sleep(1)

r = redis.StrictRedis(host="localhost", port=6379, db=0)

c = Coordinator(r)

le = preprocessing.LabelEncoder()
categories = [x[0] for x in sdm.read_features(feature_training_file, 
  names_only=True)]
le.fit(categories)

dlearn = DistributedEnsembleLearner(c, SdmInterface())
dlearn.fit(feature_training_file)

#preds = dlearn.predict("feats_test.h5")
all_preds = dlearn.predict(feature_testing_file, aggregate=False )
min_vote = MinorityVote().predict(all_preds)
maj_vote = MajorityVote().predict(all_preds)

# Find the actual classes.
fn = sdm.read_features(feature_testing_file).names
actual=get_actual(fn, le.classes_)

num_right_maj = sum([(i == j) for i,j in zip(maj_vote, actual)])
print("Majority vote sdm got {} predictions correct for an accuracy of {}."
      .format(num_right_maj, num_right_maj / len(actual)))

num_right_min = sum([(i == j) for i,j in zip(min_vote, actual)])
print("Minority vote sdm got {} predictions correct for an accuracy of {}."
      .format(num_right_min, num_right_min / len(actual)))

c.publish_shutdown()

