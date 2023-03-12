import os
import torch
from torch import nn

import pandas as pd
import numpy as np

import Util

class Discriminator(nn.Module):
    # Classifies data as real or synthetic, used to train generator
    # Consists of 3 sequential linear layers, standerdized by dropout
    # Uses a Relu activation function
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(df_dim, (batch_size*8)),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear((batch_size*8), (batch_size*4)),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear((batch_size*4), (batch_size*2)),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear((batch_size*2), 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

class Generator(nn.Module):
    # Generates synthetic data
    # Consists of 3 sequential linear layers, standerdized by dropout
    # Uses a Relu activation function
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(df_dim, (batch_size)),
            nn.ReLU(),
            nn.Linear((batch_size), (batch_size*2)),
            nn.ReLU(),
            nn.Linear((batch_size*2), df_dim),
        )

    def forward(self, x):
        output = self.model(x)
        return output

def training_loop(num_epochs, train_loader, val_loader, lr=0.002, early_stopping_patience=50):
# This is where the generator and discrimator are trained
    # loss function
    loss_function = nn.BCELoss()
    # Optimizer
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience = 0
    for epoch in range(num_epochs):

        for n, (real_samples, _) in enumerate(train_loader):

        # Data for training the discriminator
            real_samples_labels = torch.ones((batch_size, 1))
            latent_space_samples = torch.randn((batch_size, df_dim))
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size, 1))
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

        # Data for training the generator
            latent_space_samples = torch.randn((batch_size, df_dim))

        # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

    # Train on validation dataset
        with torch.no_grad():
            val_loss_discriminator = 0
            val_loss_generator = 0
            for val_samples, _ in val_loader:
                # Data for validation
                real_samples_labels_val = torch.ones((batch_size, 1))
                latent_space_samples_val = torch.randn((batch_size, df_dim))
                generated_samples_val = generator(latent_space_samples)
                generated_samples_labels_val = torch.zeros((batch_size, 1))
                all_samples_val = torch.cat((real_samples, generated_samples))
                all_samples_labels_val = torch.cat((real_samples_labels, generated_samples_labels))

                # Calculate loss on validation dataset
                output_discriminator_val = discriminator(all_samples_val)
                val_loss_discriminator += loss_function(output_discriminator_val, all_samples_labels_val).item()

                output_discriminator_generated_val = discriminator(generated_samples_val)
                val_loss_generator += loss_function(output_discriminator_generated_val, real_samples_labels_val).item()

            # Average loss on validation dataset
            val_loss_discriminator /= len(val_loader)
            val_loss_generator /= len(val_loader)
            print(f"Epoch: {epoch} Validation Loss D.: {val_loss_discriminator:.4f}, Validation Loss G.: {val_loss_generator:.4f}")
