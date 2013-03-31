import time, sdm
import cnidaria as cn
import redis
from collections import Counter
 
def majority(votes):
    C = Counter(votes)
    maj = sum(C.itervalues()) / 2.0  # possibly faster than len(votes)
    pair = C.most_common(1)[0]
    return pair[0] if pair[1] > maj else None

# Clean this up.
def majority_vote(preds):
  if len(preds) == 0:
    raise(RuntimeError("No predictions supplied"))
  ret=[]
  for i in xrange(len(preds[0])):
    votes=[]
    for j in xrange(len(preds)):
      votes.append(preds[j][i])
    ret.append(majority(votes))
  return(ret)

cn.start_local_workers(3)
time.sleep(0.5)

c = cn.Coordinator(redis_handle=redis.StrictRedis(host="localhost", port=6379,
  db=0))

# Train the ensemble of workers.
train_workers='''
import sdm, random, struct
from functools import partial

with open("/dev/random", "rb") as f:
  rnd_str = f.read(4)
  rand_int = struct.unpack('I', rnd_str)[0]
  random.seed(rand_int)

learner = sdm.sdm.SupportDistributionMachine()

features = sdm.extract_features.read_features("feats_train.h5",
  subsample_fn=partial(random.sample, k=40))

proc_features, pca, scaler = sdm.proc_features.process_features(
  features, ret_pca=True, ret_scaler=True)


y = [ (1 if f == 'country' else 0) for f in proc_features.categories]

learner.fit(proc_features.features, y)
'''

c.publish_exec(train_workers)

load_test_features='''
test_features = sdm.extract_features.read_features("feats_test.h5")

test_proc = sdm.proc_features.process_features(test_features, pca=pca,
  scaler=scaler)

actual = [ (1 if 'country' in f else 0) for f in test_proc.names]
'''

c.publish_exec(load_test_features)

# Get the predictions. 
# TODO: Dial back the number of processors being used per learner.
preds = c.publish_eval("learner.predict(test_proc.features)")
print(preds)

actual = c.publish_get("actual")[0]

mv = majority_vote(preds)
num_right = sum([(i == j) for i,j in zip(mv, actual)])
print("Ensemble sdm got "+ str(num_right) + " predictions correct "+
  "for an accuracy of "+str(float(num_right)/float(len(actual))))

print("Individual accuracies are:")
for k in range(len(preds)):
  num_right = sum([(i == j) for i,j in zip(preds[k], actual)]) 
  print("\t"+str(num_right)+" "+str(float(num_right)/float(len(actual))))

c.publish_shutdown()


