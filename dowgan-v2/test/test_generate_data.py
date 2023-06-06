"""
Test for Util.py
"""
import unittest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from data_preprocess import minmax_scaler, split_target_condition
from cgan import CGAN_Generator

import sys
import getpass
sys.path.insert(0, f'/home/{getpass.getuser()}/dowgan/dowgan')
import generate_data
df = pd.read_csv(f'/home/{getpass.getuser()}/dowgan/notebooks/ems-experiments/dataimpurity.csv')
data = df.drop(df.columns[0],axis=1)
time = np.arange(0,len(data))
data.insert(0,'Time',time)
targets, conditions = split_target_condition(data, condition='Class')
target_tensor, conditions_tensor, scaler = minmax_scaler(target_data=targets,
                                                 condition_data=conditions,
                                                 min=0,
                                                 max=1)
INPUT_SHAPE = 46
HIDDEN_UNITS = 8
OUTPUT_SHAPE_G = 45
netG = CGAN_Generator(input_shape=INPUT_SHAPE, 
                      hidden_units=HIDDEN_UNITS, 
                      output_shape=OUTPUT_SHAPE_G)
column_names = list(data.columns.values[0:45])
class Test_Name(unittest.TestCase):
    
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        generate_data.generate_samples(generator_model=netG,
                                       minmax_scaler=scaler,
                                       n_datapoints=1000,
                                       n_features=45,
                                       column_names=column_names)
        
    def test_name(self):
        """
        Test for name error, data not found
        """
        with self.assertRaises(NameError):
            generate_data.generate_samples(generator_model=netG,
                                           minmax_scaler=scaler,
                                           n_datapoints=1000,
                                           n_features=45,
                                           column_names=test)
