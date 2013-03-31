import sdm 
#import LearnerInterface

#class SdmInterface(LearnerInterface):
class SdmInterface:

  def fit_string(self, data_handle,
    subsample_fn_string="partial(random.sample, k=80)"):

    rs='''import sdm, random, struct
from functools import partial

with open('/dev/random', 'rb') as f:
  rnd_str = f.read(4)
  rand_int = struct.unpack('I', rnd_str)[0]
  random.seed(rand_int)

learner = sdm.SupportDistributionMachine()

features = sdm.read_features('DATAHANDLE',
  subsample_fn=SUBSAMPLEFN)

unique_categories = list(set(features.categories))
cat2num = dict(zip(unique_categories, range(len(unique_categories))))
num2cat = dict(zip(range(len(unique_categories)), unique_categories))

proc_features, pca, scaler = sdm.process_features(
  features, ret_pca=True, ret_scaler=True)

y = [cat2num[cat] for cat in proc_features.categories]

learner.fit(proc_features.features, y)
'''
    rs = rs.replace("DATAHANDLE", data_handle).replace("SUBSAMPLEFN", 
      subsample_fn_string)
    return(rs)

  def predict_string(self, data_handle):
    pred_string='''test_features = sdm.read_features("DATAHANDLE")

test_proc = sdm.process_features(test_features, pca=pca,
  scaler=scaler)
pred_num=learner.predict(test_proc.features)
preds = [num2cat[num] for num in pred_num]
'''
    pred_string = pred_string.replace("DATAHANDLE", data_handle)
    return(pred_string)
    
  def get_pred_string(self):
    return("preds")
