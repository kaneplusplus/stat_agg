import sdm 
#import LearnerInterface

#class SdmInterface(LearnerInterface):
class SdmInterface:

  def __init__(self, subsample_fn_string="partial(random.sample, k=80)",
    n_proc=2):
    self.subsample_fn_string=subsample_fn_string
    self.n_proc=n_proc

  def fit_string(self, data_handle):

    rs='''import sdm, random, struct
from functools import partial
from sklearn import preprocessing

with open('/dev/random', 'rb') as f:
  rnd_str = f.read(4)
  rand_int = struct.unpack('I', rnd_str)[0]
  random.seed(rand_int)

learner = sdm.SDC(n_proc=NPROC)

features = sdm.read_features('DATAHANDLE', subsample_fn=SUBSAMPLEFN)

le = preprocessing.LabelEncoder()
le.fit(features.categories)

proc_features, pca, scaler = sdm.process_features(
  features, ret_pca=True, ret_scaler=True)

learner.fit(proc_features.features, le.transform(proc_features.categories))
'''
    rs = rs.replace("DATAHANDLE", data_handle).replace("SUBSAMPLEFN", 
      self.subsample_fn_string).replace("NPROC", str(self.n_proc))
    return(rs)

  def predict_string(self, data_handle):
    pred_string='''test_features = sdm.read_features("DATAHANDLE")
test_proc = sdm.process_features(test_features, pca=pca, scaler=scaler)
preds = le.inverse_transform(learner.predict(test_proc.features))
'''
    pred_string = pred_string.replace("DATAHANDLE", data_handle)
    return(pred_string)
    
  def get_pred_string(self):
    return("preds")
