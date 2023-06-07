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
sys.path.insert(0, f'/home/{getpass.getuser()}/dowgan/dowgan')
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

df = pd.DataFrame(np.random.randint(0,100,size=(10, 3)), columns=list('ABC'))
df_empty = pd.DataFrame()
class TestDetermineComponents(unittest.TestCase):
    """
    Unit test for the function determine_components
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        util.determine_components(df)
    def test_value(self):
        """
        Test for value error, when dataframe is empty
        """
        with self.assertRaises(ValueError):
            util.determine_components(df_empty)
    def test_name(self):
        """
        Test for name error, when dataframe isn't defined
        """
        with self.assertRaises(NameError):
            util.determine_components(test)


losses = [0.7081860899925232, 0.7069265842437744, 0.6873602271080017]
ls = {'a':[1, 2, 3]}
class TestPlotLosses(unittest.TestCase):
    """
    Unit test for the function plot_losses
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        util.plot_losses(losses)
    def test_type(self):
        """
        Test for type error, when wrong type of data
        """
        with self.assertRaises(TypeError):
            util.plot_losses(ls)
    def test_name(self):
        """
        Test for name error, when data isn't defined
        """
        with self.assertRaises(NameError):
            util.plot_losses(test)
            
gen_losses = torch.tensor(losses)
dis_losses = torch.tensor(losses)
rec_losses = torch.tensor(losses)
class TestPlotMultipleLosses(unittest.TestCase):
    """
    Unit test for the function plot_multiple_losses
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        util.plot_multiple_losses([gen_losses,dis_losses,rec_losses],
                                  ['Generator Loss', 'Discriminator Loss', 'Recovery Loss'])
    def test_attribute(self):
        """
        Test for attribute error, when data isn't a tensor doens't have attribute 'detach'
        """
        with self.assertRaises(AttributeError):
            util.plot_multiple_losses([losses,dis_losses,rec_losses],
                                      ['Generator Loss', 'Discriminator Loss', 'Recovery Loss'])
    def test_name(self):
        """
        Test for name error, when data isn't defined
        """
        with self.assertRaises(NameError):
            util.plot_multiple_losses([test,dis_losses,rec_losses],
                                      ['Generator Loss', 'Discriminator Loss', 'Recovery Loss'])

array1 = np.random.rand(1,100)
array2 = np.random.rand(1,100)
array3 = np.random.rand(1,20)
name = ['x1', 'x2']
class TestPlotFeatures(unittest.TestCase):
    """
    Unit test for the function plot_features
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        util.plot_features(array1, array2, name, 2)
    def test_index(self):
        """
        Test for index error, when list index out of range
        """
        with self.assertRaises(IndexError):
            util.plot_features(array1, array2, name, 3)
    def test_name(self):
        """
        Test for name error, when data isn't defined
        """
        with self.assertRaises(NameError):
            util.plot_features(test, array2, name, 2)

array4 = np.random.rand(5,100)
array5 = np.random.rand(5,100)
class TestPlotPca(unittest.TestCase):
    """
    Unit test for the function plot_pca
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        util.plot_pca(array4, array5)
    def test_value(self):
        """
        Test for value error, when number of components doesn't fulfill solver
        """
        with self.assertRaises(ValueError):
            util.plot_pca(array1, array5)
    def test_name(self):
        """
        Test for name error, when data isn't defined
        """
        with self.assertRaises(NameError):
            util.plot_pca(test, array5)
            
class TestPlotTsne(unittest.TestCase):
    """
    Unit test for the function plot_tsne
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        util.plot_tsne(array1, array2)
    def test_value(self):
        """
        Test for value error, when input array dimensions doesn't match
        """
        with self.assertRaises(ValueError):
            util.plot_tsne(array1, array3)
    def test_name(self):
        """
        Test for name error, when data isn't defined
        """
        with self.assertRaises(NameError):
            util.plot_tsne(test, array3)
            
if __name__ == '__main__':
    unittest.main()
    
