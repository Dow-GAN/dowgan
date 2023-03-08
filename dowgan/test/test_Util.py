"""
Test for Util.py
"""
import unittest
import numpy as np
import pandas as pd
import torch
import getpass
import sys
sys.path.insert(0, f'/home/{getpass.getuser()}/dowgan/dowgan')
import Util

#load a dataset for testing
df = pd.read_csv('http://faculty.washington.edu/dacb/HCEPDB_moldata.zip').sample(5000, random_state=69)
df2 = pd.read_csv('http://faculty.washington.edu/dacb/HCEPDB_moldata.zip').sample(5000, random_state=69).drop(columns=['SMILES_str', 'tmp_smiles_str', 'stoich_str'])
#array for testing
array = np.random.rand(4, 4)
arrays = np.random.rand(4, 4, 4)

class Test_Name(unittest.TestCase):
    
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        Util.get_column_names(df)
        
    def test_attribute(self):
        """
        Test for attribute error, not a dataframe
        """
        with self.assertRaises(AttributeError):
            Util.get_column_names(array)
            
    def test_key(self):
        """
        Test for key error, no column name
        """
        with self.assertRaises(KeyError):
            Util.get_column_names(df[0])
class Test_Array(unittest.TestCase):
    
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        Util.create_arrays(df2,5000)
        
    def test_attribute(self):
        """
        Test for value error, data in some columns are strings
        """
        with self.assertRaises(ValueError):
            Util.create_arrays(df,500)
            
    def test_index(self):
        """
        num_data exceed data size
        """
        with self.assertRaises(IndexError):
            Util.create_arrays(df,5001)
class Test_Tensor(unittest.TestCase):
    
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        Util.create_tensors(array)
        
    def test_attribute(self):
        """
        Test for Runtime error, array has wrong dimension
        """
        with self.assertRaises(RuntimeError):
            Util.create_tensors(arrays)
            
    def test_key(self):
        """
        Test for key error
        """
        with self.assertRaises(KeyError):
            Util.create_tensors(df)