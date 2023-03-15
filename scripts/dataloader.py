''' Load csv data, specify parameters and train GAN Model'''
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

import platform
import getpass
import sys

# specifying a different path for mac users
if platform.system() == 'Darwin':
    sys.path.insert(0, f'/Users/{getpass.getuser()}/dowgan/dowgan')
else:
    sys.path.insert(0, f'/Home/{getpass.getuser()}/dowgan/dowgan')

import Util
import tsGAN

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_style('white')
sns.set_context('poster')

# Import Data Frame
# CSV path
folder_name = "data"
file_name = "hungary_chickenpox.csv"
path = os.path.join('../', folder_name, file_name)
#Load CSV into Dataframe
df = pd.read_csv(path,sep=',')
df = df.drop(columns = ['Date'])
# Get column names
names = Util.get_column_names(df)
# Set random state
torch.manual_seed(111)
# Select # of data points from data set to train GAN
num_data = 522
# Specify dimensionality of dataframe
df_dim = len(df.columns)
# Learning rate
lr = 0.005
# Number of epoch
num_epochs = 500
# Discriminator dropout for data standardization
drop_out = 0.2
# Define batch size for the data loader
batch_size = 18

# Create arrays and then turn into tensors for the train loader
arrays = Util.create_arrays(df[:num_data],num_data)
train_set = Util.create_tensors(arrays)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

# Define generator and discriminator
generator = tsGAN.Generator(df_dim,batch_size,drop_out)
discriminator =  tsGAN.Discriminator(df_dim,batch_size,drop_out)

# Training the model
dim_list, gen_list = tsGAN.training_loop(generator, discriminator, num_epochs, train_loader,
                                         batch_size,lr,df_dim)

# After training the Gen, generates data as tensors
latent_space_samples = torch.randn((num_data), df_dim)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()

# Takes the generated data and turns it into a pandas DataFrame
df_gen=pd.DataFrame(generated_samples)
df_gen = df_gen.set_axis(names, axis=1, inplace=False)
print(df_gen)

# Plot the generated data
fig, axes = plt.subplots()
sns.lineplot(data=df_gen,x=df_gen.index, y=df_gen[names[0]], color='#1b9e77')
axes.set_title('Generated Data - Budapest')
axes.set_xlabel("Weeks")

#sns.scatterplot(data=df_gen,x=df_gen[names[1]], y=df_gen[names[0]], color='g')
#axes.set_title('Generated Data')
#axes.set_xlabel("Time")

plt.show()

#Converting the losses from tensors into a DataFrame
#Initializing the loss lists
dim_list_new = []
gen_list_new = []

#Iterating through the lists of tensors and converting them to arrays
for i in np.arange(0,len(dim_list)):
    d = dim_list[i].detach().numpy()
    g = gen_list[i].detach().numpy()
    dim_list_new.append(d)
    gen_list_new.append(g)

#Putting these loss lists into Dataframes
dim_list_new=pd.DataFrame(dim_list_new, columns=['dim_loss'])
gen_list_new=pd.DataFrame(gen_list_new, columns=['gen_loss'])

loss = pd.concat([dim_list_new,gen_list_new],axis=1)

#Plot the loss of the discriminator and generator
fig, ax = plt.subplots()

sns.lineplot(data=loss, x=range(0, len(loss)),y='dim_loss',sort=False,
             label = 'discriminator',color='#268785')
sns.lineplot(data=loss, x=range(0, len(loss)),y='gen_loss',sort=False,
             label = 'generator',color='#4A225D')

ax.legend(fontsize = 16)
ax.set_ylabel('Loss')
ax.set_xlabel('Step')

plt.show()

# To run
# python3 dataloader.py
