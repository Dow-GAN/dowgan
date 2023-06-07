# TO RUN SCRIPT TYPE INTO TERMINAL:
# python3 run.py

''' Load csv data, specify parameters and train GAN Model'''
import os
import warnings
import platform
import getpass
import sys

import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

# specifying a different path for mac users
if platform.system() == 'Darwin':
    sys.path.insert(0, f'/Users/{getpass.getuser()}/dowgan/dowgan')
elif platform.system() == 'Linux':
    sys.path.insert(0, f'/home/{getpass.getuser()}/dowgan/dowgan')
elif platform.system() == 'Windows':
    sys.path.insert(0, f'/Home/{getpass.getuser()}/dowgan/dowgan')
else:
    print('Need modified file path to import Util.')

import util
import EncoderGAN

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_style('white')
sns.set_context('poster')

# Set the random seed for Python's built-in random module
seed = 1
random.seed(seed)

# Set the random seed for NumPy
np.random.seed(seed)

# Set the random seed for PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import Data Frame
# CSV path
folder_name = "data"
file_name = "mydata.csv"

# Load and Preprocess Data
file_path = os.path.join('../',folder_name, file_name)
data = pd.read_csv(file_path)
print("Successfully loaded data")

# Run PCA to determine the lower dimensionality of the data
components = util.determine_components(data)
print(f"Ran PCA, number of primary components: {components}")

# Group the data by the "Class" column
grouped_data = data.groupby('Pressure Condition')

# Create an empty dictionary to store the DataFrames
dfs = {}

# Iterate over each group and store the data in the dictionary
for class_, group in grouped_data:
    dfs[class_] = group.copy()

# Access the separate DataFrames
df_class_1 = dfs[0]
df_class_2 = dfs[1]

# Now that we have two different dataframes we can drop class column
df_class_1 = df_class_1.drop(columns= 'Pressure Condition')
df_class_2 = df_class_2.drop(columns= 'Pressure Condition')

# Get column names
column_names = df_class_2.columns

# Here we Scale the data
# Create an instance of StandardScaler
scaler1 = StandardScaler()
scaler2 = StandardScaler()

# Fit the scaler to data
scaler1.fit(df_class_2)
scaler2.fit(df_class_2)

# Transform the data to Standard scale
df_class_1_scaled = scaler1.transform(df_class_1)
df_class_2_scaled = scaler2.transform(df_class_2)

# train test split
train_df, test_df = util.create_train_test(df_class_2_scaled, test_size=0.2, columns = column_names)
# Parameters
#### input_dim = Input dimension, number of features in feature dimension
#### hidden_dim = Hidden dimension, size of the network internals  
#### output_dim = Output dimension, dimension of the embedded space. this is what the feature space is reduced to 
#### encoding_dim = Encoding dimension, should be that of the embedding space
#### noise_dim = Noise dimension, should be that of the embedding space
#### cond_dim = Condition dimension, should be that of the embedding space
#### seq_length = Sequence Length, length of input sequence for data loading into the model
#### input_channels = Dimension of input going into the convolutional layer in discriminator
#### hidden_channels = Size of the hidden discriminator layers
#### num_epochs = Number of training epochs for embedding and recovery networks
#### lr = Learning rate how much the weights are modified at each optimizer step
#### batch_size = batch size is the size of data used in training
#### criterion = Loss function for embedding and recovery networks
input_dim = len(column_names)  
hidden_dim = 50  
output_dim = components  
encoding_dim = output_dim
noise_dim = output_dim
cond_dim = output_dim
seq_length = 20
input_channels = seq_length + 2
hidden_channels = 64
num_epochs = 500 # Embedder Recovery
num_epochs_gan = 200 # Generator Discriminator
lr = 0.0001  
batch_size = 32

# Instantiate the Dataset, Dataloader, Embedding and Recovery Networks
dataset = EncoderGAN.TimeSeriesDataset(train_df, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
embedding_network = EncoderGAN.EmbeddingNetwork(input_dim, hidden_dim, encoding_dim)
recovery_network = EncoderGAN.RecoveryNetwork(encoding_dim, hidden_dim, input_dim)
# Train the Model
print("Begin training Embedding and Recovery Models")
losses = EncoderGAN.train_autoencoder(embedding_network, recovery_network, dataloader, input_dim, num_epochs, lr)
# Plot Loss
print("Plotting loss for Embedding and Recovery Model")
util.plot_losses(losses)

# Instantiate the models
generator = EncoderGAN.ConditionalGenerator(noise_dim, cond_dim, hidden_dim, output_dim)
discriminator = EncoderGAN.ConditionalDiscriminator(input_channels, hidden_channels, output_dim)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
print("Begin training Generator and Discriminator Models")
gen_losses, dis_losses, rec_losses = EncoderGAN.train_GAN(dataloader, embedding_network,
                                               generator, discriminator, recovery_network, generator_optimizer,
                                               discriminator_optimizer, num_epochs_gan, print_every=10)
# Plot Loss Curves
print("Plotting loss for Generator and Discriminator Models")
util.plot_multiple_losses([gen_losses,dis_losses,rec_losses], ['Generator Loss', 'Discriminator Loss', 'Recovery Loss'])

# Test Data Generation

## Subsequent conditions use the last row of the previous generated sequence
# Parameters: 
# num_samples = number of samples/sequences
# seq_length = sequence length, length of each sequence
num_samples = 8
seq_length = 50
# Specify test data
test_data_gen = test_df.iloc[400:800]
test_data_tensor = torch.from_numpy(test_data_gen.values).unsqueeze(0).float()  
# Encode and determine condition
encoded = embedding_network(test_data_tensor)
condition = encoded[:,0,:]
# generate new data and inverse transform the values
print("Generating data from single input condition")
recovered_sequences = util.generate_timeseries_sequences(generator, recovery_network, embedding_network, test_data_gen, num_samples, seq_length)
recovered_sequences = scaler2.inverse_transform(recovered_sequences)
test_data_inverse = scaler2.inverse_transform(test_data_gen)
recovered_sequences.shape, test_data_inverse.shape
#print("Saving generated data as 'generated_data'")
#file_name = 'generated_data.csv'
#util.array_to_dataframe(recovered_sequences, column_names, file_name)
print("Plot PCA and tSNE")
util.plot_pca(test_data_inverse, recovered_sequences)
util.plot_tsne(test_data_inverse, recovered_sequences)

# Test Data Augmentation

# Parameters: 
# num_samples = dimensionality of input keep = 1 
# seq_length = sequence length, length of each sequence
num_samples = 1
seq_length = 20
# Specify test data
test_data_aug = test_df.iloc[0:1400]
print("Generating data from recuring input conditions")
# generate new data and inverse transform the values
augmented_data = util.augment_timeseries_sequences(generator, recovery_network, embedding_network, test_data_aug, num_samples, seq_length)
augmented_data = scaler2.inverse_transform(augmented_data)
test_data_inv = scaler2.inverse_transform(test_data_aug)
#print("Saving generated data as 'augmented_data'")
#file_name = 'augmented_data.csv'
#util.array_to_dataframe(augmented_data, column_names, file_name)
print("Plot PCA and tSNE")
util.plot_pca(test_data_inv, augmented_data)
util.plot_tsne(test_data_inv, augmented_data)
