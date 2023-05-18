import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')

from cgan import CGAN_Discriminator, CGAN_Generator
from tqdm.auto import tqdm


def train_model(discriminator_model: CGAN_Discriminator,
                generator_model: CGAN_Generator,
                dataloader: DataLoader,
                epochs: int,
                batch_size: int,
                learning_rate: float,
                num_classes=2):
    """
    training loop for the CGAN model. returns D_loss list and G_loss list.

    inputs: discriminator model, generator model, number of epochs, 
    batch_size (should be same as dataloader), learning rate
    output: D_loss list, G_loss list
    ---
    !currently this model is designed for data with only 2 conditions
    and is measured via BCELoss
    """

    loss_D_list = []
    loss_G_list = []

    epochs = epochs
    netD = discriminator_model
    netG = generator_model

    netD.train()
    netG.train()

    # optimizer for discriminator and generator
    optimizerD = torch.optim.Adam(netD.parameters(),lr=learning_rate,betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(),lr=learning_rate,betas=(0.5, 0.999))

    real_condition = torch.ones([batch_size, 1], dtype=torch.float)
    fake_condition = torch.zeros([batch_size, 1], dtype=torch.float)

    G_loss_fn = nn.MSELoss()
    D_loss_fn = nn.BCELoss()

    # Iterating through epochs
    for epoch in tqdm(range(epochs)):

        # Iterating through batches (i: iterable int, target: target, condition: condition)
        for i, (target, condition) in enumerate(dataloader):

            # for iterating through only FULL BATCHES
            if len(condition) == batch_size:

                # initializng fixed noise to generator
                fixed_noise = torch.randn((batch_size, len(target[0]), len(target[0][0])))

                '''
                    Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                '''
                # Generating data and feeding it to the discriminator
                G_result = netG(fixed_noise,condition)
                D_fake_result = netD(G_result,condition)

                # Inputting real results into discriminator
                D_real_result = netD(target, condition)

                # Calculating the loss for the real and the generated data
                D_real_loss = D_loss_fn(D_real_result, real_condition)
                D_fake_loss = D_loss_fn(D_fake_result, fake_condition)

                # Back propagation
                D_train_loss = (D_real_loss + D_fake_loss) / 2
                netD.zero_grad()
                D_train_loss.backward()
                optimizerD.step()
                '''
                    Update G network: maximize log(D(G(z)))
                '''
                new_condition = torch.LongTensor(batch_size, len(condition[0]), 1).random_(1, num_classes)
                G_result = netG(fixed_noise, new_condition)
                D_fake_result = netD(G_result, new_condition)
                G_train_loss = G_loss_fn(D_fake_result, real_condition)

                # Back propagation
                netD.zero_grad()
                netG.zero_grad()
                G_train_loss.backward()
                optimizerG.step()
            else:
                pass

        # per epoch, append loss info:
        loss_G_list.append(G_train_loss)
        loss_D_list.append(D_train_loss)

        # print progress per 10 epochs
        if (epoch+1) % 10 == 0:
                    print(f"Epoch: {epoch+1}/{epochs} | D_loss: {D_train_loss:.4f} | G_loss: {G_train_loss:.4f}")
        else:
            pass

    return loss_D_list, loss_G_list



def get_loss(D_loss_list: list, G_loss_list: list, epochs: int):
    """
    takes the D_loss and G_loss outputs from the train_model function
    and turns them into a dataframe and returns a loss graph

    inputs: D_loss list (train_model function output 1), 
    G_loss list (train_model function output 2), 
    epochs = number of epochs used in train_model function
    """
    D_loss = []
    G_loss = []

    for i, loss in enumerate(D_loss_list):
        d = loss.detach().numpy()
        D_loss.append(d)
    for i, loss in enumerate(G_loss_list):
        g = loss.detach().numpy()
        G_loss.append(g)

    # stacking arrays
    D_loss_array = np.vstack(D_loss).flatten()
    G_loss_array = np.vstack(G_loss).flatten()

    loss_df = pd.DataFrame({
        'epochs': np.arange(1, epochs+1, epochs/len(D_loss_array)),
        'D_loss': D_loss_array,
        'G_loss': G_loss_array
    })

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=loss_df, x='epochs', y='D_loss', label='discriminator')
    sns.lineplot(data=loss_df, x='epochs', y='G_loss', label='generator')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show();