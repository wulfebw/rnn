"""
Main script for training and testing the stacked encoder-decoder rnn.

todo
(1) toy datasets
(2) ucf 101 dataset
(3) build and test rnn
(4) build and test enc_dec_rnn
(5) build and test stacked_enc_dec_rnn

"""

from enc_dec_rnn import StackedEncDecRNN
from toy_datasets import ToyVideoDataset, easy_data_generator
from ucf_101 import UCF_101

def get_dataset_toy_hard():
	trainset = ToyVideoDataset()
	testset = ToyVideoDataset()
	return trainset, testset

def get_dataset_toy_easy():
	trainset = ToyVideoDataset(easy_data_generator)
	testset = ToyVideoDataset(easy_data_generator)
	return trainset, testset

def get_raw_dataset():

def get_dataset_ucf_101():
	pass

def get_enc_dec_rnn(structure):
	# 1. build the rnn
	pass

def get_stacked_enc_dec_rnn(structure):
	# 1. build the stacked_enc_dec_rnn
	pass

def get_trainer_sgd_stacked_enc_dec_rnn(model, trainset):
	# 1. build the training algorithm
	# 2. build the Train object
	pass

def main():
	# 1. load the dataset
	# 2. build the stacked enc-dec rnn
	# 3. train the stacked enc-dec rnn
	# 4. test the stacked enc-dec rnn
	pass

if __name__ == '__main__':
	main()



