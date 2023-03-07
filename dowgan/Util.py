''' Utility functions for the Time Series GAN Model'''
import numpy as np
import pandas as pd
import torch


def get_column_names(dataframe):
    ''' Parse Column for names and put names in list'''
    column_names=[]
    for col in dataframe.columns:
            column_names.append(col)
    return column_names
    

def create_arrays(dataframe, num_data):
    ''' get the number of columns in the DataFrame'''
    # Get number of columns
    num_cols = len(dataframe.columns)
    # create an empty list to hold the arrays
    arrays = []
    # loop over the columns and assign each to a numbered array
    for i in range(num_cols):
        col_name = dataframe.columns[i]
        col_array = np.zeros(num_data)
        for j in range(num_data):
            col = dataframe[col_name].values
            col_array[j] = col[j]
        arrays.append(col_array)
    # arrays now contains a list of NumPy arrays, each corresponding to a column of the DataFrame
    return arrays


def create_tensors(arrays):
    # Creates the tensors for neural network
    num_data = len(arrays[0])
    num_arrays = len(arrays)
    train_data = torch.zeros((num_data, num_arrays))
    # Takes the arrays and turns them into tensors
    for i in range(num_arrays):
        train_data[:, i] = torch.from_numpy(arrays[i])

    # create the labels tensor
    train_labels = torch.zeros(num_data)
    # Zip the tensors and tensor labels
    train_set = [(train_data[i], train_labels[i]) for i in range(num_data)]
    return train_set
