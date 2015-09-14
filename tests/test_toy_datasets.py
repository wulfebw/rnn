import sys
sys.path.append('/Users/wulfe/Dropbox/Start/scripts/machine_learning')
from stacked_enc_dec_rnn import toy_datasets, chest_accel, sign_lang

import unittest

import numpy as np

class TestToyVideoDataset(unittest.TestCase):

	""" __init__ tests """
	def test_init_data_shape(self):
		hard_dataset = toy_datasets.ToyVideoDataset(n_classes=101, n_examples=100, n_frames=10, n_features=4096)
		actual = hard_dataset.X.shape
		expected = (100, 10, 4096)
		self.assertEquals(actual, expected)

	def test_init_data_elements(self):
		hard_dataset = toy_datasets.ToyVideoDataset(n_classes=101)
		actual = hard_dataset.y[hard_dataset.y>100] + hard_dataset.y[hard_dataset.y<0]
		expected = []
		self.assertTrue(np.array_equal(actual, expected))

	def test_init_uses_easy(self):
		easy_dataset = toy_datasets.ToyVideoDataset(toy_datasets.easy_data_generator)
		actual = set(easy_dataset.X.flatten())
		expected = set([0.,1.])
		self.assertEquals(actual, expected)

	""" iterator tests """
	def test_iterator_basic_iteration(self):
		hard_dataset = toy_datasets.ToyVideoDataset()
		actual = False
		for item in hard_dataset:
			actual = True
			break
		self.assertTrue(actual)

	def test_iterator_n_iterations(self):
		n_examples = 10
		hard_dataset = toy_datasets.ToyVideoDataset(n_examples=n_examples)
		actual = 0
		for item in hard_dataset:
			actual += 1
		expected = n_examples
		self.assertEquals(actual, expected)

class TestDataGenerators(unittest.TestCase):

	""" hard_data_generator tests """
	def test_hard_data_generator_valid_shape(self):
		n_classes, n_examples, n_frames, n_features = 101, 10, 10, 5
		X, y = toy_datasets.hard_data_generator(n_classes, n_examples, n_frames, n_features)
		actual = X.shape
		expected = (n_examples, n_frames, n_features)
		self.assertEquals(actual, expected)

	""" easy_data_generator tests """
	def test_easy_data_generator_valid_shape(self):
		n_classes, n_examples, n_frames, n_features = 101, 10, 0, 5
		X, y = toy_datasets.easy_data_generator(n_classes, n_examples, n_frames, n_features)
		actual = X.shape
		n_frames_expected = len(bin(n_classes)) - 2
		expected = (n_examples, n_frames_expected, n_features)
		self.assertEquals(actual, expected)

	""" easy_softmax_data_generator tests """
	def test_easy_softmax_data_generator_valid_data(self):
		n_classes, n_examples = 5, 10
		X, y = toy_datasets.easy_softmax_data_generator(n_classes, n_examples)
		actual = True
		for example, target in zip(X, y):
			if example[target] != 1:
				actual = False
		self.assertTrue(actual)

	def test_easy_softmax_data_generator_data_len(self):
		n_classes, n_examples = 7, 10
		X, y = toy_datasets.easy_softmax_data_generator(n_classes, n_examples)
		self.assertTrue(len(X) == len(y))

	""" chest_accel_data_generator tests """
	def test_equalize_class_distribution_valid_data_ordered(self):
		X = [[0], [1], [2], [3], [4], [5]]
		y = [1, 2, 1, 2, 1, 1]
		X, y = chest_accel.equalize_class_distribution(X, y)
		expected_X = np.array([[0], [1], [2], [3]])
		expected_y = np.array([1, 2, 1, 2])
		self.assertTrue(np.array_equal(X, expected_X))
		self.assertTrue(np.array_equal(y, expected_y))

	def test_equalize_class_distribution_valid_data_unordered(self):
		X = [[0], [1], [2], [3], [4], [5], [6]]
		y = [1, 2, 1, 2, 1, 1, 2]
		X, y = chest_accel.equalize_class_distribution(X, y)
		expected_X = np.array([[0], [1], [2], [3], [4], [6]])
		expected_y = np.array([1, 2, 1, 2, 1, 2])
		self.assertTrue(np.array_equal(X, expected_X))
		self.assertTrue(np.array_equal(y, expected_y))

	def test_equalize_class_distribution_invalid_data(self):
		X = []
		y = []
		X, y = chest_accel.equalize_class_distribution(X, y)
		self.assertTrue(np.array_equal(X, np.array([])))
		self.assertTrue(np.array_equal(y, np.array([])))

	def test_truncate_to_smallest_valid(self):
		X = np.array([[[1,2,3], [1,2,3], [2,3,3]], 
				[[3,4,5], [6,5,3]], 
				[[2,3,2], [1,4,3], [6,5,2]],
				[[1,1,1],[1,1,1],[1,1,1],[1,1,1]]])
		actual = chest_accel.truncate_to_smallest(X)
		expected = np.array([[[1,2,3], [1,2,3]], 
				[[3,4,5], [6,5,3]], 
				[[2,3,2], [1,4,3]],
				[[1,1,1],[1,1,1]]])
		self.assertTrue(np.array_equal(actual, expected))

	""" sign_lang_data_generator tests """
	def test_get_linguistic_class_label_from_filepath_valid_name(self):
		filepath = 'test/test/test/hand0.sign'
		actual = sign_lang.get_linguistic_class_label_from_filepath(filepath)
		expected = 'hand'
		self.assertEquals(actual, expected)

	def test_create_linguistic_to_numeric_dict_from_filepaths_valid_paths(self):
		filepaths = ['test/test/test/hand0.sign', 'test/test/test/hand1.sign', 'test/test/test/face0.sign',
						'test/test/test/face1.sign', 'test/test/test/smile0.sign', 'test/test/test/hand0.sign']
		actual = sign_lang.create_linguistic_to_numeric_dict_from_filepaths(filepaths)
		expected = {'hand': 0, 'face': 1, 'smile': 2}
		self.assertItemsEqual(actual, expected)

	def test_pad_data_to_max_sample_length_valid_data_3_dim(self):
		data = np.array([[[1,2,3], [1,2,3], [2,3,3]], 
				[[3,4,5], [6,5,3]], 
				[[2,3,2], [1,4,3], [6,5,2]],
				[[1,1,1],[1,1,1],[1,1,1],[1,1,1]]])
		actual = sign_lang.pad_data_to_max_sample_length(data)
		expected = np.array([[[1,2,3], [1,2,3], [2,3,3], [0,0,0]], 
				[[3,4,5], [6,5,3], [0,0,0], [0,0,0]], 
				[[2,3,2], [1,4,3], [6,5,2], [0,0,0]],
				[[1,1,1],[1,1,1],[1,1,1],[1,1,1]]])
		self.assertTrue(np.array_equal(actual, expected))






