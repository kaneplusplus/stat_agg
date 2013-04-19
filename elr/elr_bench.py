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
import argparse

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

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-w", "--num-local-workers", default=1, type=int,
    help="The number of local workers to start")
  parser.add_argument("-p", "--num-procs", default=2, type=int,
    help="The number of parallel processors per worker")
  parser.add_argument("-s", "--sample-prop", default=0.4, type=float,
    help="The proportion of training features to sample from")
  parser.add_argument("-o", "--host", default="localhost", type=str,
    help="The location of the host redis server")
  parser.add_argument("-P", "--port", default=6379, type=int,
    help="The port for the host redis server")
  parser.add_argument("-d", "--db", default=0, type=int,
    help="The redis database channel to connect to")
  parser.add_argument("-t", "--train", type=str,
    help="The training hdf5 data set")
  parser.add_argument("-l", "--validate", type=str,
    help="The validation hdf5 data set")
  parser.add_argument("-r", "--predict", type=str,
    help="The prediction hdf5 data set")
  parser.add_argument("-a", "--aggregator-type", default="var",
    type=str,
    help="The aggregator type. Valid arguemnts are minority, majority, var")
  parser.add_argument("-x", "--path", default="", type=str,
    help="The path to the python binary")
  parser.add_argument("-v", "--verbose", action="store_const", const=True,
    default=False, help="Output extra information")
  return parser.parse_args()

def main():

  args = parse_args()

  # Create a label encoder and find out how many test cases there are
  # total.
  le = preprocessing.LabelEncoder()
  categories = [x[0] for x in sdm.read_features(args.train, names_only=True)]
  le.fit(categories)

  try:
    # Start the local workers.
    # TODO: This should fixed so that when the start_local_workers function
    # returns the workers are running and waiting for tasks.
    start_local_workers(nw=args.num_local_workers, path=args.path, 
      verbose=args.verbose)
    time.sleep(0.5)

    # Create the coordinator.
    r = redis.StrictRedis(host=args.host, port=args.port, db=args.db)
    c = Coordinator(r)

    # Create the distributed ensemble learner, which is a collection of
    # sdms.
    if (args_sample.prop < 1):
      dlearn = DistributedEnsembleLearner(c, 
        SdmInterface(subsample_fn_string="partial(random.sample, k="+
          str(int(round(args.sample_prop*len(categories))))+"),", 
          n_proc=args.num_procs))
    else:
      dlearn = DistributedEnsembleLearner(c, 
        SdmInterface(subsample_fn_string="None", n_proc=args.num_procs))

        
    dlearn.fit(args.train)

    # Get the predictions for the validation data set and compare it
    # with the validation actuals to train the aggregator.
    validate_preds = dlearn.predict(args.validate)
    file_info = sdm.read_features(args.validate, names_only=True)
    validate_actuals = get_actual([x[1] for x in file_info], le.classes_)

    # Create the ensemble aggregator.
    agg_map = {"minority" : MinorityVote, "majority" : MajorityVote, 
      "var" : MinimumClassificationVariance} 
    agg = agg_map[args.aggregator_type]()
    agg.train({"actual": validate_actuals, "prediction" : validate_preds})

    # Now do the predition.
    agg_pred = agg.predict(dlearn.predict(args.predict))
    file_info = sdm.read_features(args.validate, names_only=True)
    actual = get_actual([x[1] for x in file_info], le.classes_)

    num_right = sum([(i == j) for i,j in zip(agg_pred, actual)])
    print(("Majority vote sdm got {} predictions correct out of {}"+
      " for an accuracy of {}.").format(num_right, len(actual), 
      num_right / len(actual)))

  except Exception as e:
    print("Exception: {0}".format(e.message))
  except:
    print("Unknown exception caught, shutting down local workers")

  c.publish_shutdown()

if __name__=='__main__':
  main()
