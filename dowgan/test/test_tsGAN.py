"""
Unit Test for tsGAN.py
"""
import unittest

import pandas as pd
import torch
import os
import getpass
import sys
sys.path.insert(0, f'/home/{getpass.getuser()}/dowgan/dowgan')
import tsGAN
import Util

# Import DataFrame and variables for testing
# CSV path
folder_name = "data"
file_name = "hungary_chickenpox.csv"
path = os.path.join(f'/home/{getpass.getuser()}/dowgan/dowgan', folder_name, file_name)
#Load CSV into Dataframe
df = pd.read_csv(path,sep=',')
df = df.drop(columns = ['Date'])
# Get column names
names = Util.get_column_names(df)
# Set random state
torch.manual_seed(111)
# Select # of data points from data set to train GAN
num_data = 450
# Specify dimensionality of dataframe
df_dim = len(df.columns)
bad_dim = 10
# Learning rate
lr = 0.002
# Number of epoch
num_epochs = 100
# Discriminator dropout for data standardization
drop_out = 0.2
# Define batch size for the data loader
batch_size = 25
bad_batch = 33
# Create arrays and then turn into tensors for the train loader
arrays = Util.create_arrays(df[:num_data],num_data)
train_set = Util.create_tensors(arrays)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

# Define generator and discriminator
generator = tsGAN.Generator(df_dim,batch_size,drop_out)
discriminator =  tsGAN.Discriminator(df_dim,batch_size,drop_out)
#training_loop(generator, discriminator, num_epochs, train_loader, batch_size, lr, df_dim)

class TestTrainingLoop(unittest.TestCase):
    """
    Unit test for the function get_column_name
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        tsGAN.training_loop(generator, discriminator, num_epochs, train_loader,
                            batch_size, lr, df_dim)
    def test_value(self):
        """
        Test for value error, when num_data % batch_size != 0
        """
        with self.assertRaises(ValueError):
            tsGAN.training_loop(generator, discriminator, num_epochs, train_loader,
                                bad_batch, lr, df_dim)
    def test_runtime(self):
        """
        Test for runtime error, wrong df_dim value causing the tensor size to be wrong
        """
        with self.assertRaises(RuntimeError):
            tsGAN.training_loop(generator, discriminator, num_epochs, train_loader,
                                batch_size, lr, bad_dim)
