"""

:development:

	:plan:
		(1) write some functions to load the data
			(a) for each signer-session, go through the directory collecting the sequences
				(i) each sequence should have the format [[features at t=0], [features at t=1], ...] class label
		(2) write all that data to one file in the above format so we don't have to load every time
			(a) numpy.save() and numpy.load() should be used
		(3) write a function to load it from that file and standardize the sign lenghts
			(a) mean length is 51, max is 102
			(b) maybe the MaskSequenceSpace does just this, but what if we make every sample len 102 by padding with zeros, and then stop the fprop early if the padding is reached?
			(c) works for me 
		(4) alter the lstm code to account for padding in the data

	:todo:
		(0) or just try it out as is
		(1) save a version of the padded data

"""

import re
import os
import csv
import sys
import glob
import operator
import collections

import numpy as np

import chest_accel

def pad_data_to_max_sample_length(X, pad_value=0.):
	if len(X) <= 1:
		return X

	sample_unit_shape = np.shape(X[0][0])
	pad_unit = np.tile(pad_value, sample_unit_shape)

	max_sample_len = max(map(len, X))
	new_X = []
	masks = []
	for sample in X:
		mask = np.ones_like(sample)
		pad_len = max_sample_len - len(sample)
		pad_arr = np.tile(pad_unit, (pad_len, 1))
		new_X.append(np.vstack((sample, pad_arr)))
		masks.append(np.vstack((mask, pad_arr)))

	return np.array(new_X), np.array(masks)

def load_data_from_aggregate_file(input_filepath='/Users/wulfe/Dropbox/Start/scripts/machine_learning/stacked_enc_dec_rnn/data/sign_lang.npz'):
	arrays = np.load(input_filepath)
	X = arrays['X']
	y = arrays['y']
	return X, y

def write_data_to_aggregate_file(output_filepath, X, y):
	np.savez(output_filepath, X=X, y=y)
	
def get_linguistic_class_label_from_filepath(input_filepath):
	basename = os.path.splitext(os.path.basename(input_filepath))[0]
	basename_without_numeric = re.sub(r'\d+', '', basename)
	return basename_without_numeric

def create_linguistic_to_numeric_dict_from_filepaths(filepaths):
	linguistic_to_numeric_dict = dict()
	numeric_label = 0
	for f in filepaths:
		class_label = get_linguistic_class_label_from_filepath(f)
		if class_label not in linguistic_to_numeric_dict:
			linguistic_to_numeric_dict[class_label] = numeric_label
			numeric_label += 1
	return linguistic_to_numeric_dict

def load_raw_data_from_file(input_filepath):
	data = []
	with open(input_filepath, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			data.append(map(float, row[:10]))
	return np.array(data)

def median_filter(sample):
	return sample

def load_data_from_file(input_filepath, linguistic_to_numeric_dict):
	"""
	:description: loads the data from a single sign sequence and formats it as X, y; where X = [[features at t=0], [features at t=1], ...] and y = class label (int). Rember that this is a single sample.
	"""
	# 1. load in the raw data
	sample = load_raw_data_from_file(input_filepath) 

	# 2. get the linguistic class label from the filepath
	linguistic_class_label = get_linguistic_class_label_from_filepath(input_filepath)

	# 3. retrieve the numeric class label from the linguistic_to_numeric_dict
	numeric_class_label = linguistic_to_numeric_dict[linguistic_class_label]

	# 4. optionally apply median filter 
	sample = median_filter(sample)

	return sample, numeric_class_label

def get_filepaths_from_dir(dir):
	filepaths = []
	for dirpath, dirnames, files in os.walk(dir):
		for f in files:
			if f.endswith('.sign'):
				filepaths.append(os.path.join(dirpath, f))
	return filepaths


def load_data_from_dir(data_dir='/Users/wulfe/Downloads/signs'):
	# filepaths = get_filepaths_from_dir(data_dir)
	filepaths = get_filepaths_from_dir(data_dir)
	print('potential number of samples: {}'.format(len(filepaths)))
	linguistic_to_numeric_dict = create_linguistic_to_numeric_dict_from_filepaths(filepaths)
	print('len(linguistic_to_numeric_dict): {}'.format(len(linguistic_to_numeric_dict)))
	print(sorted(linguistic_to_numeric_dict.items(), key=operator.itemgetter(1)))
	X = []
	y = []
	for f in filepaths:
		print('loading file: {}'.format(f))
		sample, target = load_data_from_file(f, linguistic_to_numeric_dict)
		if len(sample) > 30 and len(sample) < 80:
			X.append(sample)
			y.append(target)
	print('actual number of samples: {}'.format(len(X)))
	return X, y

if __name__ == '__main__':
	data_dir = '/Users/wulfe/Downloads/signs'
	X, y = load_data_from_dir(data_dir)

	output_filepath = '/Users/wulfe/Dropbox/Start/scripts/machine_learning/stacked_enc_dec_rnn/data/sign_lang.npz'
	write_data_to_aggregate_file(output_filepath, X, y)

	X, y = load_data_from_aggregate_file(output_filepath)
	print(X.shape)
	X = chest_accel.truncate_to_smallest(X)
	#X = pad_data_to_max_sample_length(X)
	print(X.shape)
	print(y.shape)

