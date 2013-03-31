from SdmInterface import SdmInterface
from DistributedEnsembleLearner import DistributedEnsembleLearner
from DistributedEnsembleLearner import vote
from DistributedEnsembleLearner import tally
from cnidaria import *
import redis, time, sdm

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

# Start the workers and train them.
start_local_workers(nw=5)
time.sleep(1)

# Create the the handle to redis.
r = redis.StrictRedis(host="localhost", port=6379, db=0)

# Feed it to the coordinator.
c = Coordinator(r)
# ... yum

# Get the classes from the training features file.

features = \
  list(set(sdm.read_features("feats_train.h5").categories))

dlearn = DistributedEnsembleLearner(c, SdmInterface())
dlearn.fit("feats_train.h5")

#preds = dlearn.predict("feats_test.h5")
all_preds = dlearn.predict("feats_test.h5", aggregate=False )
min_vote = vote(all_preds, partial(tally, majority=False))
maj_vote = vote(all_preds, partial(tally, majority=True))

# Find the actual classes.
fn = sdm.read_features("feats_test.h5").names
actual=get_actual(fn, features)

num_right_maj = sum([(i == j) for i,j in zip(maj_vote, actual)])
print("Marjoity vote sdm got "+ str(num_right_maj) + " predictions correct "+
  "for an accuracy of "+str(float(num_right_maj)/float(len(actual))))

num_right_min = sum([(i == j) for i,j in zip(min_vote, actual)])
print("Minority vote sdm got "+ str(num_right_min) + " predictions correct "+
  "for an accuracy of "+str(float(num_right_min)/float(len(actual))))

c.publish_shutdown()

