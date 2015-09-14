"""
:filename: chest_accel.py

:description: Data loading utilities for the chest accelerometer dataset: 

	https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer#

"""

import csv
import sys
import glob
import collections

import numpy as np

def load_data_from_file(input_filepath):
	"""
	:description: load data from a single csv file into X and y components
	"""
	data = []
	try:
		with open(input_filepath, 'r') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for row in reader:
				if row[-1] != '0':
					data.append(map(float, row))
	except IOError as e:
		raise IOError('input filename: {} raised IOError on read'.format(input_filename))

	data = np.array(data)
	# X should exclude the first and last values (should have three per timestep)
	X = data[:,1:-1]
	# y is just the last value aka target
	# want 0-based indexing so subtract 1 from each target value
	# target expected to be int so cast it as such
	y = map(int, data[:,-1] - 1)
	return X, y

def truncate_to_smallest(X):
	min_len = min(map(len, X))
		
	new_X = []
	for item in X:
		new_X.append(item[:min_len])

	return np.array(new_X)


def format_as_timeseries(X, y, n_timesteps):
	"""
	:description: format the data into samples of time length n_timesteps
	"""
	n_samples = X.shape[0] / n_timesteps
	n_truncate = X.shape[0] % n_timesteps
	n_features = X.shape[1]
	new_X = np.array(X[:-n_truncate]).reshape((n_samples, n_timesteps, n_features))
	new_y = y[::n_timesteps]
	new_y = new_y[:n_samples]
	return new_X, new_y

def format_as_behavior(X, y):
	"""
	:description: the actual data is segmented into relatively long time lengths by different behaviors. This function separates the data into these relatively long behaviors in the from of X and y components.
	"""
	indicies = []
	prev_y = y[0]
	new_y = [y[0]]
	for idx, cur_y in enumerate(y):
		if cur_y != prev_y:
			indicies.append(idx)
			new_y.append(cur_y)
			prev_y = cur_y
	new_X = np.array(np.split(X, indicies))
	return new_X, np.array(new_y)

def equalize_class_distribution(X, y):
	counter = collections.Counter(y)
	if len(counter.values()) < 1:
		return X, y

	# print('class counts: {}\n'.format(counter.values()))


	max_occurences = min(counter.values())
	new_X = []
	new_y = []
	y_counts = collections.defaultdict(lambda: 0)
	for example, target in zip(X, y):
		y_counts[target] += 1
		if y_counts[target] <= max_occurences:
			new_X.append(example)
			new_y.append(target)

	counter = collections.Counter(new_y)
	# print('class counts: {}\n'.format(counter.values()))

	return np.array(new_X), np.array(new_y)


def load_data_from_dir(data_dir='/Users/wulfe/Downloads/Activity Recognition from Single Chest-Mounted Accelerometer', n_timesteps=20, reverse=False):
	files = glob.glob('{}/*.csv'.format(data_dir))
	if reverse:
		files = files[::-1]
	X = []
	y = []
	for idx, f in enumerate(files):
		print('loading {}'.format(f))
		if idx > 20:
			break
		new_X, new_y = load_data_from_file(f)
		if X == []:
			X = new_X
			y = new_y
		else:
			X = np.row_stack((X, new_X))
			y = np.concatenate((y, new_y))
	
	X = np.array(X)	
	y = np.array(y)
	
	#X, y = format_as_timeseries(X, y, n_timesteps)
	X, y = format_as_behavior(X, y)
	X = truncate_to_smallest(X)
	X, y = equalize_class_distribution(X, y)
	return X, y

if __name__ == '__main__':
	data_dir = '/Users/wulfe/Downloads/Activity Recognition from Single Chest-Mounted Accelerometer'
	n_timesteps = 20
	X, y = load_data_from_dir(data_dir, n_timesteps)
	print(X.shape)
	print(y.shape)
	for sample, target in zip(X, y):
		print(sample.shape)
		print(target)



		
