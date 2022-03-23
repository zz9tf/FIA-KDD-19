import numpy as np
import sys
sys.path.append("..")
from influence.dataset import DataSet

def load_movielens(train_dir, validation_size=5000):

  train = np.loadtxt("%s/ml-1m-ex.train.rating"%train_dir, delimiter='\t')
  valid = np.loadtxt("%s/ml-1m-ex.valid.rating"%train_dir, delimiter='\t')
  test = np.loadtxt("%s/ml-1m-ex.test.rating"%train_dir, delimiter='\t')

  train_input = train[:975460,:2].astype(np.int32)
  train_output = train[:975460,2]
  valid_input = valid[:-6, :2].astype(np.int32)
  valid_output = valid[:-6, 2]
  test_input = test[:-6, :2].astype(np.int32)
  test_output = test[:-6, 2]
  # test_input = test[:, :2].astype(np.int32)
  # test_output = test[:, 2]

  train = DataSet(train_input, train_output)
  validation = DataSet(valid_input, valid_output)
  test = DataSet(test_input, test_output)

  return {"train": train, "validation": validation, "test": test}


  
