''' Model Functions for Encoder Time Series GAN Model'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import util


class EmbeddingNetwork(nn.Module):
    """
    A PyTorch Module that represents an embedding network.

    This module takes in sequences of a specified dimension, processes them with a Gated Recurrent Unit (GRU) layer,
    then passes the output through a linear layer and a sigmoid activation function to produce embedded sequences.

    Attributes:
        gru (nn.GRU): The GRU layer of the network.
        linear (nn.Linear): A linear layer of the network.
        sigmoid (nn.Sigmoid): The Sigmoid activation function.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmbeddingNetwork, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        The forward propagation method for the network.

        Args:
            x (torch.Tensor): The input tensor. Shape should be (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: The output tensor after being processed by the network. Shape is (batch_size, seq_len, output_dim).
        """
        out, _ = self.gru(x)
        out = self.linear(out)
#         out = self.sigmoid(out)
        return out

class RecoveryNetwork(nn.Module):
    """
    A PyTorch Module that represents a recovery network.

    This module takes in sequences of a specified dimension, processes them with a Gated Recurrent Unit (GRU) layer,
    then passes the output through a linear layer to produce recovered sequences.

    Attributes:
        gru (nn.GRU): The GRU layer of the network.
        linear (nn.Linear): A linear layer of the network.
    """
    def __init__(self, output_dim, hidden_dim, input_dim):
        super(RecoveryNetwork, self).__init__()
        self.gru = nn.GRU(output_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        The forward propagation method for the network.

        Args:
            x (torch.Tensor): The input tensor. Shape should be (batch_size, seq_len, output_dim).

        Returns:
            torch.Tensor: The output tensor after being processed by the network. Shape is (batch_size, seq_len, input_dim).
        """
        out, _ = self.gru(x)
        out = self.linear(out)
        return out


class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset for handling time series data.

    This class is designed to create subsequences of a specified length from a given time series. These subsequences
    can then be used as input for sequence-to-sequence prediction tasks, such as forecasting or anomaly detection.

    Attributes:
        data (torch.Tensor): The entire time series, converted to a PyTorch tensor.
        seq_length (int): The length of the subsequences that will be produced.
    """
    def __init__(self, data, seq_length):
        """
        Initializes the TimeSeriesDataset instance.

        Args:
            data (pd.DataFrame or np.ndarray): The entire time series. Can be a pandas DataFrame or a numpy array.
            seq_length (int): The length of the subsequences that will be produced.
        """
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        """
        Returns the number of subsequences that can be produced from the time series.

        Returns:
            int: The number of subsequences.
        """
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, index):
        """
        Returns the subsequence starting at a specific index in the time series.

        Args:
            index (int): The start index for the subsequence.

        Returns:
            torch.Tensor: The subsequence of length seq_length starting at the given index.
        """
        return self.data[index:index+self.seq_length]

class ConditionalGenerator(nn.Module):
    """
    A conditional generator in a Generative Adversarial Network (GAN).

    This generator takes a noise tensor and a condition tensor as input and generates
    synthetic data that is conditioned on the given condition.

    Attributes:
        noise_layer (nn.Linear): Transforms the noise tensor into the hidden space.
        condition_layer (nn.Linear): Transforms the condition tensor into the hidden space.
        gru (nn.GRU): A GRU unit that processes the combined noise and condition tensor.
        fc (nn.Linear): Transforms the GRU output into the output space.
    """
    def __init__(self, noise_dim, cond_dim, hidden_dim, output_dim):
        super().__init__()
        self.noise_layer = nn.Linear(noise_dim, hidden_dim)
        self.condition_layer = nn.Linear(cond_dim, hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim*2, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, noise, condition):
        """
        Forward pass of the conditional generator.

        Args:
            noise (torch.Tensor): The noise tensor.
            condition (torch.Tensor): The condition tensor.

        Returns:
            torch.Tensor: The generated synthetic data.
        """
        batch_size, seq_len, _ = noise.size()
        noise_embedding = F.relu(self.noise_layer(noise.view(batch_size * seq_len, -1)))
        noise_embedding = noise_embedding.view(batch_size, seq_len, -1)
        condition_embedding = F.relu(self.condition_layer(condition))
        condition_embedding = condition_embedding.unsqueeze(1).repeat(1, seq_len, 1)
        # Adjust the batch size of condition_embedding to match noise_embedding
        # Adjust the batch size of condition_embedding to match noise_embedding
        if len(condition_embedding.shape) == 2:  # condition tensor has a batch dimension
            if condition_embedding.size(0) < batch_size:
                condition_embedding = condition_embedding.repeat(batch_size, 1, 1)
        elif len(condition_embedding.shape) == 1:  # condition tensor doesn't have a batch dimension
            condition_embedding = condition_embedding.unsqueeze(0).repeat(batch_size, 1, 1)

        combined = torch.cat([noise_embedding, condition_embedding], dim=2)
        output, _ = self.gru(combined)
        output = self.fc(output)

        return torch.tanh(output)
    
class ConditionalDiscriminator(nn.Module):
    """
    A conditional discriminator in a Generative Adversarial Network (GAN).

    This discriminator takes real or synthetic data and a condition tensor as input and outputs
    a score indicating the "realness" of the data given the condition.

    Attributes:
        conv1 (nn.Conv1d): The first convolutional layer.
        pool1 (nn.MaxPool1d): The first pooling layer.
        conv2 (nn.Conv1d): The second convolutional layer.
        pool2 (nn.MaxPool1d): The second pooling layer.
        fc (nn.Linear): The final linear layer.
    """
    def __init__(self, input_channels, hidden_channels, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels*2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # dummy input to compute flattening size
        x = torch.zeros(1, input_channels, output_dim)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        self.flattened_size = x.view(1, -1).shape[1]

        self.fc = nn.Linear(in_features=self.flattened_size, out_features=1)

    def forward(self, x, condition):
        """
        Forward pass of the conditional discriminator.

        Args:
            x (torch.Tensor): The input data, which can be real or synthetic.
            condition (torch.Tensor): The condition tensor.

        Returns:
            torch.Tensor: The score indicating the "realness" of the data given the condition.
        """
        x = torch.cat([x, condition.unsqueeze(1)], dim=1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = torch.sigmoid(self.fc(x))

        return x

    
def train_autoencoder(embedding_network, recovery_network, dataloader, input_dim, num_epochs, lr):
    """
    Train the autoencoder consisting of an EmbeddingNetwork and a RecoveryNetwork.

    The autoencoder is trained using the Adam optimizer and the MSE loss function. The loss is calculated by comparing
    the output of the autoencoder with the original input. The goal is to minimize the reconstruction error.

    Args:
        embedding_network (nn.Module): The encoder part of the autoencoder.
        recovery_network (nn.Module): The decoder part of the autoencoder.
        dataloader (DataLoader): The dataloader that provides the training data.
        input_dim (int): The dimensionality of the input data.
        num_epochs (int): The number of epochs for training.
        lr (float): The learning rate for the Adam optimizer.

    Returns:
        list: The losses at the end of each epoch.
    """
    losses = []

    criterion = nn.MSELoss()
    embedding_optimizer = torch.optim.Adam(embedding_network.parameters(), lr=lr, weight_decay=1e-5)
    recovery_optimizer = torch.optim.Adam(recovery_network.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(num_epochs):
        for seqs_in in dataloader:

            # Ensure the seqs are of correct size
            seqs_in = seqs_in.view(seqs_in.size(0), -1, input_dim)

            # Zero the gradients
            embedding_optimizer.zero_grad()
            recovery_optimizer.zero_grad()

            # Pass the input sequences through the autoencoder
            encoded = embedding_network(seqs_in)
            decoded = recovery_network(encoded)

            # Calculate the loss
            loss = criterion(decoded, seqs_in)  # The goal is to reconstruct the input sequences

            # Backward propagation
            loss.backward()

            # Update the weights
            embedding_optimizer.step()
            recovery_optimizer.step()

        losses.append(loss.item())

        # Print loss for every 10th epoch
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    return losses


def train_GAN(dataloader, embedding_network, generator, discriminator, recovery_network, generator_optimizer, discriminator_optimizer, num_epochs=3, print_every=10):
    """
    Train a conditional generative adversarial network (GAN).

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        embedding_network (nn.Module): Embedding network to encode the input data.
        generator (nn.Module): Generator network to generate synthetic data.
        discriminator (nn.Module): Discriminator network to distinguish real and synthetic data.
        recovery_network (nn.Module): Recovery network to reconstruct the input from the generated data.
        generator_optimizer (torch.optim.Optimizer): Optimizer for the generator.
        discriminator_optimizer (torch.optim.Optimizer): Optimizer for the discriminator.
        num_epochs (int, optional): Number of epochs to train for. Default is 3.
        print_every (int, optional): How often to print the loss. Default is every 10 epochs.

    Returns:
        list: List of generator losses.
        list: List of discriminator losses.
        list: List of reconstruction losses.
    """
    gen_losses = []
    dis_losses = []
    rec_losses = []
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Encode the data
            encoded = embedding_network(batch)
            condition = encoded[:,0,:]

            # Generate fake data
            noise = torch.randn(encoded.shape)
            generated = generator(noise, condition)

            # Discriminator loss
            real_data = torch.cat((encoded, condition.unsqueeze(1)), dim=1)  # Add extra dimension to condition
            fake_data = torch.cat((generated, condition.unsqueeze(1)), dim=1)
            real_preds = discriminator(real_data, condition)
            fake_preds = discriminator(fake_data.detach(), condition)  # Detach so we don't backprop through G

            real_loss = F.binary_cross_entropy(real_preds, torch.ones_like(real_preds))
            fake_loss = F.binary_cross_entropy(fake_preds, torch.zeros_like(fake_preds))
            discriminator_loss = (real_loss + fake_loss) / 2

            # Backprop and update D weights
            discriminator.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()
    
            # Generator loss
            fake_preds = discriminator(fake_data, condition)
            # Changed to Mean Squared Error loss
            generator_loss = F.mse_loss(fake_preds, torch.ones_like(fake_preds))

            # Backprop and update G weights
            generator.zero_grad()
            generator_loss.backward(retain_graph=True)
            generator_optimizer.step()

            # Decode the generated data and compute reconstruction loss
            recovery = recovery_network(generated)
            # Compute MSE between the recovery and the original data
            rec_loss = F.mse_loss(recovery, batch)
        rec_losses.append(rec_loss)
        gen_losses.append(generator_loss)
        dis_losses.append(discriminator_loss)
        
        if epoch % print_every == 0:
            print(f'Epoch {epoch}/{num_epochs}, '
                  f'Generator Loss: {generator_loss.item():.5f}, '
                  f'Discriminator Loss: {discriminator_loss.item():.5f}, '
                  f'Reconstruction Loss: {rec_loss.item():.5f}')

    return gen_losses, dis_losses, rec_losses
