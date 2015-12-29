import time
from statagg import *
from pandas import read_csv, DataFrame
from statistics import pstdev, variance
from pdb import set_trace

# Download the data and create the learners.
# from subprocess import call
# call(["Rscript", "make_data.r"]) 

rp = read_csv("regression_predictions_train.csv")

learner_names = rp.columns.values[:10]
preds={}
for ln in learner_names:
  preds[ln] = rp[ln]

pred_data = {"prediction" : preds, "actual" : rp["actual"]}

mco = MinimumContinuousOLS()
train_start = time.time()
mco.train(pred_data)
train_stop = time.time()

print("%.2f" % (train_stop - train_start) + " seconds for training.")

rp = read_csv("regression_predictions_test.csv")
learner_names = rp.columns.values[:10]
ols_agg = DataFrame()

for i in range(1, 11):
  preds={}
  for ln in learner_names[:i]:
    preds[ln] = rp[ln]
  #set_trace()
  test_start=time.time()
  ols_agg["learners_"+str(i)] = mco.predict(preds)
  test_stop=time.time()
  print("%.2f" % (test_stop - test_start) + " seconds to predict " +
        str(i) + " variables.")

ols_agg["actual"] = rp["actual"]

ols_agg.to_csv("ols_pred.csv")

