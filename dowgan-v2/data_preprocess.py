import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

# making a custom dataset class
class ImpuritiesDataset(Dataset):
    def __init__(self, target, conditions):
        self.target = target
        self.conditions = conditions
    def __len__(self):
        return len(self.conditions)
    def __getitem__(self, idx):
        condition = self.conditions[idx]
        target = self.target[idx]
        return target, condition
        
def split_target_condition(data, condition: str):
    '''split given dataframe into a target dataframe
        and condition dataframe
        input: all of the data and a single column that is the condition
        output: df of conditions data and df of target data'''
    
    conditions = data.loc[:,condition]
    targets = data.loc[:, ~data.columns.isin(['Class'])]
    
    print(f"shape of targets (all features without operating conditions): {targets.shape}")
    print(f"shape of conditions (operating conditions): {conditions.shape}")

    return targets, conditions
    
def minmax_scaler(target_data,condition_data, min: float, max: float):
    '''Scales data and transforms target and condition data to tensors
        input: target data, condition data, min for scaling, max for scaling
        output: a tensor of the target data (scaled) and tensor of conditions data (unscaled)'''
    scaler = MinMaxScaler(feature_range=(min,max))

    # fit & transform scaler data
    targets = scaler.fit_transform(target_data)

    column_names = list(target_data.columns.values)
    target_data_scaled = pd.DataFrame(targets, columns=column_names)

    target_tensor = torch.tensor(target_data_scaled.values, dtype=torch.float32)
    conditions_tensor = torch.tensor(condition_data, dtype=torch.float32).unsqueeze(dim=1)
    
    return conditions_tensor, target_tensor

def data_batch(target_tensor,conditions_tensor,n_datapoints: int, batch_size:int):
    '''batching data of conditions and target tensors
        put into dataloader'''
    n_samples = len(target_tensor)//n_datapoints
    
    target_tensor_list = []
    conditions_tensor_list = []
    invalid_batch_indexes = []

    #Iterating through all of the data and dividing it up
    for i in np.arange(0, n_samples, 1):
        
        #Taking a batch of the target data
        temp_target_batch = torch.split(target_tensor, n_datapoints, dim=0)[i]
        
        #Checking for any NaNs in that tensor
        # if its true, not saving it and the condition tensor to the list
        if torch.isnan(temp_target_batch).any() == True:
            invalid_batch_indexes.append(i)
            continue
        else:
            target_tensor_list.append(temp_target_batch)

        #Taking the corresponding batch of conditions
        temp_condition_batch = torch.split(conditions_tensor, n_datapoints, dim=0)[i]
        conditions_tensor_list.append(temp_condition_batch)

    print(f'length of data: {target_tensor.shape[0]}')
    print(f'We have {len(target_tensor_list)} samples with {n_datapoints} datapoints in each.')
    print(f'There are {len(invalid_batch_indexes)} invalid batches.\n')

    #Inputting data into ImpuritiesDataset class
    train_data = ImpuritiesDataset(target=target_tensor_list,
                               conditions=conditions_tensor_list)
    
    train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=False,
    )

    print(f"dataloader: {train_dataloader}")
    print(f"length of train_dataloader: {len(train_dataloader)} batches of {batch_size} (batch_size) examples")
    print(f"number of total examples: {len(train_data)} \neach example containing... \nTARGET shape (number of timepoints): {train_data[0][0].shape} \nCONDITION shape (class 1 or 2): {train_data[0][1].shape}")

    return train_dataloader

def data_batch_clean(target_tensor,conditions_tensor,n_datapoints: int, batch_size:int):
    '''ASSUMES NO NAN VALUES. Batching data of conditions and target tensors
        put into dataloader'''
    n_samples = len(target_tensor)//n_datapoints
    
    target_tensor_list = []
    conditions_tensor_list = []

    #Iterating through all of the data and dividing it up
    for i in np.arange(0, n_samples, 1):
        
        #Taking a batch of the target data
        temp_target_batch = torch.split(target_tensor, n_datapoints, dim=0)[i]
        target_tensor_list.append(temp_target_batch)

        #Taking the corresponding batch of conditions
        temp_condition_batch = torch.split(conditions_tensor, n_datapoints, dim=0)[i]
        conditions_tensor_list.append(temp_condition_batch)

    print(f'length of data: {target_tensor.shape[0]}')
    print(f'We have {len(target_tensor_list)} samples with {n_datapoints} datapoints in each.')

    #Inputting data into ImpuritiesDataset class
    train_data = ImpuritiesDataset(target=target_tensor_list,
                               conditions=conditions_tensor_list)
    
    train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=False,
    )

    print(f"dataloader: {train_dataloader}")
    print(f"length of train_dataloader: {len(train_dataloader)} batches of {batch_size} (batch_size) examples")
    print(f"number of total examples: {len(train_data)} \neach example containing... \nTARGET shape (number of timepoints): {train_data[0][0].shape} \nCONDITION shape (class 1 or 2): {train_data[0][1].shape}")

    return train_dataloader

def read_csv_drop(file_path: str,columns_to_drop: []):
    '''reads time-series data, drops the date column and any rows with NaNs.
        input: file path (str)'''
    df = pd.read_csv(file_path, sep=",")
    df = df.drop(columns=columns_to_drop)
    print(f'dataframe shape after dropping time column: {df.shape}')
    df = df.dropna()
    print(f'dataframe shape after dropping nan rows: {df.shape}')
    
    return df