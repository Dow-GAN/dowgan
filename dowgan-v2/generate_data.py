import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')

import torch

from sklearn.preprocessing import MinMaxScaler

from cgan import CGAN_Generator

def generate_samples(generator_model: CGAN_Generator,
                     minmax_scaler: MinMaxScaler,
                     n_datapoints: int,
                     n_features: int,
                     column_names: list):
    """
    generate samples based on the trained generator

    input: generator model, 
    n_datapoints (number of timepoints desired), 
    n_features (same as target columns),
    column names (list)
    output: plot of generated data
    """
    input_noise = torch.rand(1, n_datapoints, n_features)
    input_condition = torch.zeros(1, n_datapoints, 1)

    generated_samples = generator_model(input_noise, input_condition)
    generated_data_array = generated_samples.detach().numpy().squeeze()

    scaler = minmax_scaler
    generated_data_unscaled = scaler.inverse_transform(generated_data_array)
    generated_data_df = pd.DataFrame(generated_data_unscaled, columns=column_names)
    print(f"generated_data: {generated_data_df.shape}")

    plt.figure(figsize=(12,8))
    sns.lineplot(data=generated_data_df[0:10000])
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -1.35), ncols=2)
    plt.title('Generated Features vs. Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Features')
    plt.show();