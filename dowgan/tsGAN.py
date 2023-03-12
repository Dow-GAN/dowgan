''' Model Functions for Time Series GAN Model'''
import torch
from torch import nn

class Discriminator(nn.Module):
    ''' Classifies data as real or synthetic, used to train generator'''
    # Consists of 3 sequential linear layers, standerdized by dropout
    # Uses a Relu activation function
    def __init__(self,df_dim,batch_size,drop_out):
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
    ''' Generates synthetic data'''
    # Consists of 3 sequential linear layers, standerdized by dropout
    # Uses a Relu activation function
    def __init__(self,df_dim,batch_size,drop_out):
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

def training_loop(generator, discriminator, num_epochs, train_loader, batch_size, lr, df_dim):
    '''This is where the generator and discrimator are trained'''
    # Define Loss function
    loss_function = nn.BCELoss()
    # Define Optimizer for generator and discriminator
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
    # Main training loop
    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator
            # Concatenate real and fake data with apporpriate labels
            real_samples_labels = torch.ones((batch_size, 1))
            latent_space_samples = torch.randn((batch_size, df_dim))
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size, 1))
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
        # Training the discriminator
            # Zero the gradient and load concatenated data into discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            # Calculate and backpropogate loss
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()
        # Data for training the generator
            # Generate Noise
            latent_space_samples = torch.randn((batch_size, df_dim))
        # Training the generator
            # Zero out the generator gradient and feed it noise
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            # Calculate and backpropogate loss
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()
        # Show loss
            if epoch % 10 == 0 and n == batch_size - 1:
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")
