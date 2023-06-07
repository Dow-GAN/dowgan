"""
Test for util.py
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from matplotlib.testing.compare import compare_images
from matplotlib import pyplot as plt
import torch
import getpass
import sys
import util

class TestPlotSequences(unittest.TestCase):
    def setUp(self):
        self.embedding_network = MagicMock()
        self.recovery_network = MagicMock()

        self.embedding_network.return_value = torch.randn(1, 100)
        self.recovery_network.return_value = torch.randn(1, 100)
        
        self.df = pd.DataFrame(data=torch.randn(100).numpy())

    @patch('matplotlib.pyplot.show')
    def test_plot_sequences(self, mock_show):
        """
        Test verifies whether the `plot_sequences` function calls the `show` method of `matplotlib.pyplot`
        to attempt showing the plot.

        The test case creates a DataFrame (`df`) from randomly generated data and calls the `plot_sequences`
        function with the DataFrame, embedding_network, and recovery_network. It then asserts that the `show`
        method of `matplotlib.pyplot` was called exactly once.
        """
        util.plot_sequences(self.df, self.embedding_network, self.recovery_network)
        
        # Check that plot was attempted to be shown
        mock_show.assert_called_once()


class TestAugmentTimeseriesSequences(unittest.TestCase):
    def setUp(self):
        self.generator = MagicMock()
        self.recovery_network = MagicMock()
        self.embedding_network = MagicMock()

        self.num_samples = 10
        self.seq_length = 20

        # Create mock tensors to be returned by generator, recovery_network and embedding_network
        mock_gen_output = torch.randn(self.num_samples, self.seq_length)
        mock_recovery_output = torch.randn(self.num_samples, self.seq_length)
        mock_embedding_output = torch.randn(1, self.seq_length, self.num_samples)  # Adjust the shape according to your needs

        self.generator.return_value = mock_gen_output
        self.recovery_network.return_value = mock_recovery_output
        self.embedding_network.return_value = mock_embedding_output

        self.test_data = pd.DataFrame(data=np.random.rand(100, 20))

    def test_augment_timeseries_sequences(self):
        """
        Test verifies the shape of the output and checks whether the `generate_sequences` function is
        called the expected number of times.

        The test case creates a DataFrame (`test_data`) from randomly generated data and calls the
        `augment_timeseries_sequences` function with the generator, recovery_network, embedding_network,
        test_data, num_samples, and seq_length. It then asserts the shape of the result and checks that
        the `generate_sequences` function was called the expected number of times.
        """
        with patch('util.generate_sequences') as mock_generate_sequences:
            mock_generate_sequences.return_value = np.random.rand(self.seq_length, self.num_samples)
            result = util.augment_timeseries_sequences(self.generator, self.recovery_network, self.embedding_network, self.test_data, self.num_samples, self.seq_length)

            # Verify the shape of the output
            self.assertEqual(result.shape, (100, self.num_samples))

            # Verify that the generate_sequences was called the expected number of times
            self.assertEqual(mock_generate_sequences.call_count, len(self.test_data) // self.seq_length)


class TestGenerateTimeseriesSequences(unittest.TestCase):
    def setUp(self):
        self.generator = MagicMock()
        self.recovery_network = MagicMock()
        self.embedding_network = MagicMock()

        self.seq_length = 50
        self.num_samples = 8

        self.test_data = pd.DataFrame(data=np.random.rand(400, 10))
        self.test_data_tensor = torch.from_numpy(self.test_data.values).unsqueeze(0).float()

        self.condition = torch.randn(1, 10)
        
        self.mock_gen_output = torch.randn(self.num_samples, self.seq_length, 10)
        self.generator.return_value = self.mock_gen_output
        self.recovery_network.return_value = self.mock_gen_output
        self.embedding_network.return_value = self.condition.unsqueeze(0)

    def test_generate_timeseries_sequences(self):
        """
        Test verifies the shape of the output and checks whether the `generator`, `recovery_network`,
        and `embedding_network` functions are called.

        The test case creates a DataFrame (`test_data`) from randomly generated data and calls the
        `generate_timeseries_sequences` function with the generator, recovery_network, embedding_network,
        test_data, num_samples, and seq_length. It then asserts the shape of the result and checks that
        the `generator`, `recovery_network`, and `embedding_network` functions were called.
        """
        result = util.generate_timeseries_sequences(
            self.generator, self.recovery_network, self.embedding_network, self.test_data, self.num_samples, self.seq_length)
        
        self.assertEqual(result.shape, (self.num_samples * self.seq_length, 10))
        self.generator.assert_called()
        self.recovery_network.assert_called()
        self.embedding_network.assert_called()

class TestGenerateSequences(unittest.TestCase):
    def setUp(self):
        self.generator = MagicMock()
        self.recovery_network = MagicMock()

        self.seq_length = 50
        self.num_samples = 8

        self.condition = torch.randn(1, 10)
        
        self.mock_gen_output = torch.randn(self.num_samples, self.seq_length, 8)
        self.generator.return_value = self.mock_gen_output
        self.recovery_network.return_value = self.mock_gen_output

    def test_generate_sequences(self):
        """
        Test verifies the shape of the output and checks whether the `generator`, `recovery_network`,
        and `embedding_network` functions are called.

        The test case creates a DataFrame (`test_data`) from randomly generated data and calls the
        `generate_timeseries_sequences` function with the generator, recovery_network, embedding_network,
        test_data, num_samples, and seq_length. It then asserts the shape of the result and checks that
        the `generator`, `recovery_network`, and `embedding_network` functions were called.
        """
        result = util.generate_sequences(
            self.generator, self.recovery_network, self.num_samples, self.seq_length, self.condition)
        
        self.assertEqual(result.shape, (self.seq_length, self.num_samples))
        self.generator.assert_called()
        self.recovery_network.assert_called()

        
if __name__ == '__main__':
    unittest.main()
    
