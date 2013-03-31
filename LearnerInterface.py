
class LearnerInterface:
  "Interface for using learners from outside packages in the ensemble."
  def train_string(self, *args):
    "Returns the string to train remote learners"
    pass
  def predict_string(self, *args):
    "Returns the string to perform prediction on remote learners"
    pass
