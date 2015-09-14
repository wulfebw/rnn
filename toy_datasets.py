"""
Toy video datasets

let's base these off of the TIMIT sequence dataset here: https://github.com/vdumoulin/research/blob/master/code/pylearn2/datasets/timit.py

:development:
	(1) you can't create a shared var of a matrix of differing length vectors, though there are some alternative. One is by using a mask.

:questions: 
	(1) how does the iterator use the data_specs?
	(2) at what point are the features actually labeled with the string 'features'? Does this ever even occur?
"""

import sys

import numpy as np
import functools

import theano

from pylearn2.datasets.dataset import Dataset
from pylearn2.utils.iteration import FiniteDatasetIterator, resolve_iterator_class
from pylearn2.sandbox.rnn.utils.iteration import SequenceDatasetIterator
from pylearn2.sandbox.rnn.space import SequenceSpace, SequenceDataSpace
from pylearn2.space import VectorSpace, CompositeSpace, VectorSequenceSpace

import chest_accel

def hard_data_generator(n_classes, n_examples, n_frames, n_features):
	rng = np.random.RandomState(seed=42)
	X = rng.normal(size=(n_examples, n_frames, n_features))
	y = np.random.randint(0, n_classes, (n_examples, 1))
	return X, y

def hard_data_generator_reconstruct(n_classes, n_examples, n_frames, n_features):
	rng = np.random.RandomState(seed=42)
	X = rng.normal(size=(n_examples, n_frames, n_features))
	y = rng.normal(size=(n_examples, n_frames, n_features))
	return X, y

def hard_data_generator_random_reconstruct(n_classes, n_examples, n_frames, n_features):
	rng = np.random.RandomState(seed=42)
	X = rng.normal(size=(n_examples, n_frames, n_features))
	return X, X

def easy_data_generator(n_classes, n_examples, n_frames, n_features):
	rng = np.random.RandomState(seed=42)
	y = np.random.randint(0, n_classes, (n_examples))
	# n_frames is the maximum number of binary digits needed to represent a class label
	n_frames = len(bin(n_classes)) - 2
	X = []
	for target in y:
		example = []
		binary = '{0:0{1}b}'.format(int(target), n_frames)
		for char in binary:
			if char == '0':
				example.append(np.zeros((n_features)))
			else:
				example.append(np.ones((n_features)))
		X.append(example)
	X = np.array(X)
	return X, y

def easy_data_generator_reconstruct(n_classes, n_examples, n_frames, n_features):
	rng = np.random.RandomState(seed=42)
	y = np.random.randint(0, n_classes, (n_examples, 1))
	# n_frames is the maximum number of binary digits needed to represent a class label
	n_frames = len(bin(n_classes)) - 2
	X = []
	for target in y:
		example = []
		binary = '{0:0{1}b}'.format(int(target), n_frames)
		for char in binary:
			if char == '0':
				example.append(np.zeros((n_features)))
			else:
				example.append(np.ones((n_features)))
		X.append(example)
	X = np.array(X)
	return X, X

def super_easy_data_generator_reconstruct(n_classes, n_examples, n_frames, n_features):
	X = np.ones((n_examples, n_frames, n_features))
	return X, X

def easy_softmax_data_generator(n_classes, n_examples, n_frames=None, n_features=None):
	"""
	:description: returns one hot vectors that exactly correspond to the target values.

	example: y = [1,2] X = [[0,1,0], [1,0,0] ]
	"""
	y = np.random.randint(0, n_classes + 1, (n_examples))
	X = np.zeros((n_examples, n_classes + 1))
	for example, target in zip(X, y):
		example[target] = 1
	return X, y

def chest_accel_data_loader(n_classes, n_examples, n_frames, n_features):
	return chest_accel.load_data_from_dir()

def get_sequence_lengths(data):
	return np.array([len(sample) for sample in data])

class ToyVideoDataset(Dataset):
	"""
	Class for toy video datasets.

	:description: This class represents a toy video dataset. That is, a set of examples where each example is a sequence of values where those values may be lists. The class uses a passed in function to generate the data. 

		Data in the format (examples, time, data) such that:

		>>> X[0,:,:]
		gives the first example 
		>>> X[0,0,:]
		gives the first time step of the first example
		>>> X[0,0,0]
		gives the first feature from the first time step of the first example

	:development: 
	"""
	def __init__(self, data_generator=None, n_classes=101, n_examples=10, n_frames=10, n_features=4096):
		"""
		:type data_generator: function
		:param data_generator: function used to generate data in the form of X, y tuple. X is a 3-dimensional array with dimensions (examples, frames/time, features). y is a 2-dimensional array with dimensions (examples, target values). Optional value defaults to generating random therefore 'hard' data.

		:type n_classes: int
		:param n_classes: the number of possible target values or n_classes

		:type n_examples: int
		:param n_examples: the number of examples to be generated in the dataset

		:type n_frames: int
		:param n_frames: the number of frames or time steps in each example

		:type n_features: int
		:param n_features: the number of features in each time step
		"""
		rng = np.random.RandomState(seed=42)
		self.n_features = n_features
		self.n_examples = n_examples
		if data_generator is None:
			data_generator = hard_data_generator
		self.data_generator = data_generator
		self.X, self.y = self.data_generator(n_classes, n_examples, n_frames, n_features)

		features_space = VectorSequenceSpace(dim=self.n_features)
		# features_space = SequenceDataSpace(VectorSpace(dim=self.n_features))

		targets_space = VectorSequenceSpace(dim=1)
		# targets_space = SequenceDataSpace(VectorSpace(dim=1))

		space_components = [features_space, targets_space]
		space = CompositeSpace(space_components)

		source = ('features', 'targets')

		self.data_specs = (space, source)

		self._iter_mode = resolve_iterator_class('shuffled_sequential')
		self._iter_data_specs = (CompositeSpace((features_space, targets_space)), source)

	def get_data_specs(self):
		return self.data_specs

	def get_num_examples(self):
		return self.n_examples

	def get_data(self):
		return self.X, self.y

	def get_data_as_shared(self):
		shared_x = theano.shared(np.asarray(self.X, dtype=theano.config.floatX), borrow=True)
		shared_y = theano.shared(self.y, borrow=True)
		return shared_x, shared_y


	def get(self, source, index):
		"""
		:description: the iterator object repeatedly calls this 'get' function with different indicies.

		:type source: list of strings or string
		:param source: this is passed in b/c X and y are generally referred to as 'sources' and retrieved that way. The below approach is really a workaround to allow for approriately indexing X (or might be a hack that comes back to get me)
		"""
		# index[0] for X b/c index is itself a list, and using it as it results in the returned value from X being 1 dimension greater than it should be. I am not sure why index is a list.
		# does this need to change?
		rval = (self.X[index[0]], self.y[index])
		

		# print('X.shape: {}'.format(self.X.shape))
		# print('y.shape: {}'.format(self.y.shape))
		# print('X[index]: {}'.format(self.X[index]))
		# print('y[index]: {}'.format(self.y[index]))
		return rval

	@functools.wraps(Dataset.iterator)
	def iterator(self):
		"""
		:description: returns an iterator object over this dataset

		"""
		data_specs = self._iter_data_specs
		mode = self._iter_mode
		batch_size = 1
		return FiniteDatasetIterator(self, mode(self.n_examples, batch_size, None, None), data_specs=data_specs)
