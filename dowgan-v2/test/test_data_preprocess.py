"""
Unit Test for data_preprocess.py
"""
import unittest

import numpy as np
import pandas as pd
import getpass
import sys
from sklearn.preprocessing import MinMaxScaler
sys.path.insert(0, f'/home/{getpass.getuser()}/dowgan/dowgan-v2')
import data_preprocess

df = pd.read_csv(f'/home/{getpass.getuser()}/dowgan/notebooks/ems-experiments/dataimpurity.csv')
data = df.drop(df.columns[0],axis=1)
time = np.arange(0,len(data))
data.insert(0,'Time',time)
targets, conditions = data_preprocess.split_target_condition(data, condition='Class')
#Constants for testing
N_TIMEPOINTS = 24
BATCH_SIZE = 20
target_tensor, conditions_tensor, scaler = data_preprocess.minmax_scaler(target_data=targets,
                                                 condition_data=conditions,
                                                 min=0,
                                                 max=1)

class TestSplitTargetCondition(unittest.TestCase):
    """
    Unit test for the function split_target_condition
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        data_preprocess.split_target_condition(data, condition='Class');
    def test_key(self):
        """
        Test for key error, when column doesn't exist in dataframe
        """
        with self.assertRaises(KeyError):
            data_preprocess.split_target_condition(data, condition='test')
    def test_name(self):
        """
        Test for name error, when dataframe doesn't exist
        """
        with self.assertRaises(NameError):
            data_preprocess.split_target_condition(test, condition='Class')

class TestMinmaxScaler(unittest.TestCase):
    """
    Unit test for the function minmax_scaler
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        data_preprocess.minmax_scaler(target_data=targets,
                                                 condition_data=conditions,
                                                 min=0,
                                                 max=1)
    def test_value(self):
        """
        Test for value error, when Min of desired feature range is larger than max
        """
        with self.assertRaises(ValueError):
            data_preprocess.minmax_scaler(target_data=targets,
                                                 condition_data=conditions,
                                                 min=0,
                                                 max=-1)
    def test_value(self):
        """
        Test for value error, when shape of the array is incorrect
        """
        with self.assertRaises(ValueError):
            data_preprocess.minmax_scaler(target_data=conditions,
                                                 condition_data=conditions,
                                                 min=0,
                                                 max=1)

class TestDataBatch(unittest.TestCase):
    """
    Unit test for the function data_batch
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        data_preprocess.data_batch(target_tensor=target_tensor,
                        conditions_tensor=conditions_tensor,
                        n_datapoints=N_TIMEPOINTS,
                        batch_size=BATCH_SIZE)
    def test_attribute(self):
        """
        Test for attribute error, when didn't run minmax_scaler
        """
        with self.assertRaises(AttributeError):
            data_preprocess.data_batch(target_tensor=targets,
                        conditions_tensor=conditions_tensor,
                        n_datapoints=N_TIMEPOINTS,
                        batch_size=BATCH_SIZE)
    def test_value(self):
        """
        Test for value error, when 0 batch size
        """
        with self.assertRaises(ValueError):
            data_preprocess.data_batch(target_tensor=targets,
                        conditions_tensor=conditions_tensor,
                        n_datapoints=N_TIMEPOINTS,
                        batch_size=0)

class TestDataBatchClean(unittest.TestCase):
    """
    Unit test for the function data_batch_clean
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        data_preprocess.data_batch_clean(target_tensor=target_tensor,
                        conditions_tensor=conditions_tensor,
                        n_datapoints=N_TIMEPOINTS,
                        batch_size=BATCH_SIZE)
    def test_attribute(self):
        """
        Test for attribute error, when didn't run minmax_scaler
        """
        with self.assertRaises(AttributeError):
            data_preprocess.data_batch_clean(target_tensor=targets,
                        conditions_tensor=conditions_tensor,
                        n_datapoints=N_TIMEPOINTS,
                        batch_size=BATCH_SIZE)
    def test_value(self):
        """
        Test for value error, when 0 batch size
        """
        with self.assertRaises(ValueError):
            data_preprocess.data_batch_clean(target_tensor=targets,
                        conditions_tensor=conditions_tensor,
                        n_datapoints=N_TIMEPOINTS,
                        batch_size=0)
            
class TestReadCsvDrop(unittest.TestCase):
    """
    Unit test for the function read_csv_drop
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        data_preprocess.read_csv_drop(f'/home/{getpass.getuser()}/dowgan/notebooks/ems-experiments/dataimpurity.csv',
                                      columns_to_drop=['Class'])
    def test_key(self):
        """
        Test for key error, when column doesn't exist in dataframe
        """
        with self.assertRaises(KeyError):
            data_preprocess.read_csv_drop(f'/home/{getpass.getuser()}/dowgan/notebooks/ems-experiments/dataimpurity.csv',
                                          columns_to_drop=['test'])
    def test_filenotfound(self):
        """
        Test for file not found error, when data file doesn't exist
        """
        with self.assertRaises(FileNotFoundError):
            data_preprocess.read_csv_drop('/home/whast/dowgan/notebooks/ems-experiments/test.csv',
                                          columns_to_drop=['Class'])
