from subprocess import call
from statagg import *
from pandas import read_csv, DataFrame
from statistics import pstdev, variance


# Download the data and create the learners.
# call(["Rscript", "make_data.r"]) 

rp = read_csv("regression_predictions_train.csv")

learner_names = rp.columns.values[:10]
preds={}
for ln in learner_names:
  print ln
  preds[ln] = rp[ln]

pred_data = {"prediction" : preds, "actual" : rp["actual"]}

mco = MinimumContinuousOLS()
mco.train(pred_data)

rp = read_csv("regression_predictions_test.csv")
learner_names = rp.columns.values[:10]
preds={}
for ln in learner_names:
  print ln
  preds[ln] = rp[ln]

ols_agg = DataFrame(mco.predict(preds))
ols_agg.to_csv("ols_pred.csv")

cp = read_csv("classification_predictions_train.csv")

preds={}
learner_names = cp.columns.values[:10]
for ln in learner_names:
  print ln
  preds[ln] = cp[ln]

pred_data = {"prediction" : preds, "actual" : cp["actual"]}

mcv = MinimumClassificationVariance()
mcv.train(pred_data)

cp = read_csv("classification_predictions_test.csv")

preds={}
learner_names = cp.columns.values[:10]
for ln in learner_names:
  print ln
  preds[ln] = cp[ln]

mcv_agg = DataFrame(mcv.predict(preds))
mcv_agg.to_csv("class_pred.csv")

