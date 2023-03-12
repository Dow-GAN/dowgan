''' Load csv data, spoecify parameters and train GAN Model'''
import numpy as np
import pandas as pd
import torch
from torch import nn
import Util

import tsGAN
import os

import matplotlib.pyplot as plt

import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import Data Frame
# CSV path
folder_name = "data"
file_name = "hungary_chickenpox.csv"
path = os.path.join(folder_name, file_name)
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
# Learning rate
lr = 0.002
# Number of epoch
num_epochs = 100
# Discriminator dropout for data standardization
drop_out = 0.2
# Define batch size for the data loader
batch_size = 25

# Create arrays and then turn into tensors for the train loader 
arrays = Util.create_arrays(df[:num_data],num_data)
train_set = Util.create_tensors(arrays)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

# Define generator and discriminator 
generator = tsGAN.Generator(df_dim,batch_size,drop_out)
discriminator =  tsGAN.Discriminator(df_dim,batch_size,drop_out)

# Training the model
tsGAN.training_loop(generator, discriminator, num_epochs, train_loader, batch_size,lr,df_dim)

# After training the Gen, generates data as tensors
latent_space_samples = torch.randn((num_data), df_dim)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()

# Takes the generated data and turns it into a pandas DataFrame
df_gen=pd.DataFrame(generated_samples)
df_gen = df_gen.set_axis(names, axis=1, inplace=False)
print(df_gen)
    
# Plot the generated data
fig, axes = plt.subplots(figsize=(15, 6))
sns.scatterplot(data=df_gen,x=df_gen[names[1]], y=df_gen[names[0]], color='g')
axes.set_title('Generated Data')
axes.set_xlabel("Time")
plt.show()

# To run
# python3 dataloader.py