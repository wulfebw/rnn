"""
:description: training a recurrent neural network for classification task

:development: 

	:todo:
		(1) adaptive learning rate!!!

	:options:
		(1) try to build a model that kind of works, which is to say, get this running on a ec2 instance gpu
		(2) try to implement momentum (or some other model training improvement)
			(a) stochastic grad instead of literal grad descent
		(3) implement variable len sequences
		(4) implement encoder-decoder

	
	:reminders:
		(1) the main function is currently main_theano_chest_accel, where the layers var determines the model and the task is classification

	:works:
		(1) main_theano_softmax works conclusively

	:results:

		NOTE: all these results so far are due to the fact that the 2nd index (3rd class) is more common than the others (it is the 'standing' category). As such, all these are pretty much meaningless; however, it seems that the model does seem to be working, it's just that this data is not well suited for analysis by rnn. So, get some other data.

		:chest accel:

			:1*recurrent + softmax: 

				:n_hid = *1, 1/5 data, n_epochs = 20, drop = .0:

					:accuracy: 
					:final cost: 1.93

				:n_hid = *2, all data, n_epochs = 20, drop = .0:

					:accuracy: 
					:final cost: 1.97

			:2*recurrent + softmax:

				:n_hid = *1, all data, n_epochs = 20, drop = .0:

					:accuracy: 
					:final cost: 1.85

			:1*lstm + softmax:

				:n_hid = *2, all data, n_epochs = 20, forget_gate_bias = 0.2, drop = .0:

					:accuracy: 
					:final cost: 1.91

				:n_hid = *1, all data, n_epochs = 20, forget_gate_bias = 0.05, drop = .0:

					:accuracy: 
					:final cost: 1.88

				:n_hid = *1, all data, n_epochs = 20, forget_gate_bias = 0.05, drop = .3:

					:accuracy: 
					:final cost:

			:3*lstm + softmax:

				:n_hid = *1, all data, n_epochs = 20, forget_gate_bias = 0.05, drop = .0:

					:accuracy: 
					:final cost: 1.85

		:sign_lang:

			:1*lstm + softmax:

				:n_hid = *1, all data, n_epochs = 20, forget_gate_bias = 0.05, drop = .3:

					:accuracy: 3.6%
					:final cost: 4.02


"""

import sys
sys.path.insert(1,'/Library/Python/2.7/site-packages')


import numpy as np

import theano
import theano.tensor as T

# from pylearn2.sandbox.rnn.space import SequenceSpace, SequenceDataSpace
# from pylearn2.space import VectorSpace, CompositeSpace, VectorSequenceSpace
# from pylearn2.costs.mlp import Default
# from pylearn2.sandbox.rnn.models.rnn import Recurrent, RNN
# from pylearn2.models.mlp import Softmax, MLP
# from pylearn2.training_algorithms.sgd import SGD
# from pylearn2.termination_criteria import EpochCounter
# from pylearn2.blocks import StackedBlocks
# from pylearn2.datasets.transformer_dataset import TransformerDataset
# from pylearn2.train import Train

from toy_datasets import ToyVideoDataset, easy_data_generator, hard_data_generator_reconstruct, hard_data_generator_random_reconstruct, easy_data_generator_reconstruct, super_easy_data_generator_reconstruct, chest_accel_data_loader, easy_softmax_data_generator, get_sequence_lengths
from enc_dec_rnn import EncDecRNN, EncDecRecurrent, Softmax, LSTM
from file_utils import save_model, load_model

import encoding_recurrent
import encoding_recurrent_overlap
import sign_lang
import chest_accel
import variable_length_sequence_lstm


MAX_EPOCHS_SUPERVISED = 2

def get_dataset_toy(data_generator=None, n_examples=10, n_classes=101, n_features=4096, n_frames=1):
	trainset = ToyVideoDataset(data_generator=data_generator, n_examples=n_examples, n_classes=n_classes, n_features=n_features, n_frames=n_frames)
	testset = ToyVideoDataset(data_generator=data_generator, n_examples=n_examples, n_classes=n_classes, n_features=n_features, n_frames=n_frames)
	return trainset, testset

def get_dataset_toy_raw(data_generator=None, n_examples=100, n_classes=10, n_features=20, n_frames=6, only_train=False):
	trainset, testset = get_dataset_toy(data_generator, n_examples, n_classes, n_features, n_frames)
	return trainset.get_data_as_shared(), testset.get_data_as_shared()

# def get_rnn(structure):
# 	recurrent_layer = Recurrent(dim=structure[0][1], layer_name='recurrent', irange=0.02, indices=[-1])
# 	softmax_layer = Softmax(n_classes=structure[1][1], layer_name='softmax', irange=0.02)
# 	layers = [recurrent_layer, softmax_layer]
# 	# return RNN(input_space=SequenceDataSpace(VectorSpace(dim=structure[0][0])), layers=layers)
# 	return RNN(input_space=SequenceSpace(VectorSpace(dim=structure[0][0])), layers=layers)

# def get_trainer(model, trainset):
# 	# 1. create the model config
# 	config = {'learning_rate': 0.1,
# 			'cost' : Default(),
# 			'batch_size': 1,
# 			'monitoring_batches': 1,
# 			'monitoring_dataset': trainset,
# 			'termination_criterion': EpochCounter(max_epochs=MAX_EPOCHS_SUPERVISED),
# 			'update_callbacks': None
# 			}

# 	# 2. create the training algo
# 	train_algo = SGD(**config)

# 	# 3. create and return the Train object
# 	return Train(model=model, dataset=trainset, algorithm=train_algo, extensions=None)

def get_random_indices(max_index, samples_per_epoch):
	return np.random.randint(0, max_index, samples_per_epoch)

# def main_pylearn():
# 	"""
# 	:description:
# 	:development: Instead of this layers approach, you may be able to use a MLP or RNN class to wrap the layers. This is probably what is being emmulated with this approach.
# 	"""

# 	# create datasets
# 	print('loading dataset...')
# 	n_input_at_each_timestep = 5
# 	n_classes = 6
# 	n_frames = 5
# 	trainset, testset = get_dataset_toy(n_classes=n_classes, n_features=n_input_at_each_timestep, n_frames=n_frames)

# 	print('trainset.X.shape: {}'.format(trainset.X.shape))
# 	print('type(trainset.X): {}'.format(type(trainset.X)))
# 	iter = trainset.iterator()
# 	next = iter.next()
# 	print('iter.next(): {}'.format(next))
# 	print('iter.next()[0].shape: {}'.format(next[0].shape))
# 	print('type(iter.next()): {}'.format(type(next)))
# 	print('type(iter.next()[0]): {}'.format(type(next[0])))


# 	# determine network structure
# 	n_hidden  = 5
# 	structure = [[n_input_at_each_timestep, n_hidden], [n_hidden, n_classes]]

# 	# create layers of network
# 	print('building model...')
# 	rnn = get_rnn(structure)

# 	# create the supervised trainer for the network
# 	print('building trainer...')
# 	layer_trainer = get_trainer(rnn, trainset)

# 	# run the main loop of the trainer
# 	print('training model...')
# 	layer_trainer.main_loop()

def main_theano():
	n_input_at_each_timestep = 8
	n_classes = 6
	n_frames = 4
	n_examples = 50
	print('loading data...')
	data_generator = super_easy_data_generator_reconstruct
	trainset, testset = get_dataset_toy_raw(data_generator=data_generator, n_examples=n_examples, n_classes=n_classes, n_features=n_input_at_each_timestep, n_frames=n_frames)
	trainset_X, trainset_y = trainset
	# print('trainset_X.eval(): {}'.format(trainset_X.eval()))
	# print('trainset_X.eval()[0]: {}'.format(trainset_X.eval()[0]))
	# print('trainset_X.eval()[1]: {}'.format(trainset_X.eval()[1]))
	# print('trainset_y.eval(): {}'.format(trainset_y.eval()))
	# print('trainset_y.eval()[0]: {}'.format(trainset_y.eval()[0]))
	# print('trainset_y.eval()[1]: {}'.format(trainset_y.eval()[1]))
	# sys.exit(1)

	index = T.lscalar()
	x = T.matrix('x')
	target = T.matrix('target')

	print('building model...')
	layers = [EncDecRecurrent(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep/2)]
	rnn = EncDecRNN(layers, return_indices=[-1])

	cost, updates = rnn.get_cost_updates((x, target))

	print('building trainer...')
	trainer = theano.function(
		[index],
		[cost],
		updates=updates,
		givens={
			x: trainset_X[index],
			target: trainset_y[index]
		},
		mode='FAST_COMPILE'
	)

	print('training model...')
	n_epochs = 20
	for epoch in range(n_epochs):
		costs = []
		for sample_idx in range(n_examples):
			costs.append(trainer(sample_idx)[0])
		print('training cost for epoch {0}: {1}'.format(epoch, np.mean(costs)))
	print('finished training')

	for layer in rnn.layers:
		for param in layer.params:
			print('{}: {}'.format(param.name, param.get_value()))

def main_theano_chest_accel():

	n_input_at_each_timestep = 3

	print('loading data...')
	n_classes = 6
	n_features = n_input_at_each_timestep
	data_generator = chest_accel_data_loader
	trainset, testset = get_dataset_toy_raw(data_generator=data_generator, 
											n_examples=None,
											n_classes=n_classes,
											n_features=n_features,
											n_frames=None,
											only_train=True)
	split_idx = int(round(.7 * trainset[0].shape.eval()[0]))
	print('split_idx: {}'.format(split_idx))
	trainset = trainset[0][:split_idx], trainset[1][:split_idx]
	trainset_X, trainset_y = trainset
	testset = testset[0][split_idx:], testset[1][split_idx:]
	testset_X, testset_y = testset

	index = T.lscalar()
	x = T.matrix('x')
	target = T.lscalar('target')
	print_x = theano.printing.Print('\nx')(x)
	print_target = theano.printing.Print('target')(target)

	print('building model...')
	rec_n_hid = n_input_at_each_timestep
	# layers = [EncDecRecurrent(n_vis=n_input_at_each_timestep, n_hid=rec_n_hid, return_indices=[-1]), Softmax(n_vis=rec_n_hid, n_classes=n_classes)]
	
	# single layer
	layers = [EncDecRecurrent(n_vis=n_input_at_each_timestep, n_hid=rec_n_hid, layer_name='recurrent',return_indices=[-1]), Softmax(n_vis=rec_n_hid, n_classes=n_classes)]

	# double layer
	layers = [EncDecRecurrent(n_vis=n_input_at_each_timestep, n_hid=rec_n_hid, layer_name='rec_1'), EncDecRecurrent(n_vis=rec_n_hid, n_hid=n_input_at_each_timestep, layer_name='rec_2',return_indices=[-1]), Softmax(n_vis=n_input_at_each_timestep, n_classes=n_classes)]

	# lstm
	layers = [LSTM(n_vis=n_input_at_each_timestep, n_hid=rec_n_hid, layer_name='lstm', return_indices=[-1], dropout_prob=0.3), Softmax(n_vis=rec_n_hid, n_classes=n_classes)]

	# 3*lstm
	# layers = [LSTM(n_vis=n_input_at_each_timestep, n_hid=rec_n_hid, layer_name='lstm_1'),
	# 		LSTM(n_vis=n_input_at_each_timestep, n_hid=rec_n_hid, layer_name='lstm_2'),
	# 		LSTM(n_vis=n_input_at_each_timestep, n_hid=rec_n_hid, layer_name='lstm_3', return_indices=[-1]), 
	# 		Softmax(n_vis=rec_n_hid, n_classes=n_classes)]

	# rnn = EncDecRNN(layers, cost=model_cost, return_indices=[-1])
	cost = Softmax.negative_log_likelihood
	rnn = EncDecRNN(layers, cost=cost, return_indices=[-1])

	cost, updates = rnn.get_cost_updates((x, print_target))

	print('building trainer...')
	trainer = theano.function(
		[index],
		[cost],
		updates=updates,
		givens={
			x: trainset_X[index],
			target: trainset_y[index]
		},
		mode='FAST_RUN'
	)

	errors = rnn.layers[-1].errors(target)
	validate_model = theano.function(
		inputs=[index],
		outputs=[cost, errors],
		givens={
			x: testset_X[index],
			target: testset_y[index]
		},
		mode='FAST_RUN'
	)

	print('training model...')
	n_train_examples = trainset_X.shape.eval()[0]
	n_test_examples = testset_X.shape.eval()[0]

	n_epochs = 20
	for epoch in range(n_epochs):
		costs = []
		for sample_idx in range(n_train_examples):
			costs.append(trainer(sample_idx)[0])
		print('training cost for epoch {0}: {1}\n\n'.format(epoch, np.mean(costs)))

		predictions = []
		if False:
			print('\nvalidation')
			for sample_idx in range(n_test_examples):
				predictions.append(validate_model(sample_idx)[1])
			accuracy = (1 - np.mean(predictions)) * 100
		 	print('accuracy for epoch {0}: {1}%'.format(epoch, accuracy))

	# print('finished training, final stats:\nfinal cost: {0}\naccuracy: {1}%'.format(np.mean(costs), accuracy))
	print('finished training, final stats:\nfinal cost: {0}'.format(np.mean(costs)))

	for layer in rnn.layers:
		for param in layer.params:
			print('{}: {}'.format(param.name, param.get_value()))

def main_theano_sign_lang():

	print('loading data...')
	n_input_at_each_timestep = 10
	n_classes = 97	# no base 0 considered, there are just 98 of them. May need to be 97
	dataset_sequence_length = 31
	
	X, y = sign_lang.load_data_from_aggregate_file()
	X = chest_accel.truncate_to_smallest(X)
	split_idx = int(.8 * X.shape[0])

	X = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)
	y = theano.shared(y, borrow=True)
	
	trainset_X, trainset_y = X[:split_idx], y[:split_idx]
	testset_X, testset_y = X[split_idx:], y[split_idx:]

	index = T.lscalar()
	x = T.matrix('x')
	target = T.lscalar('target')
	print_x = theano.printing.Print('\nx')(x)
	print_target = theano.printing.Print('target')(target)

	print('building model...')
	# layers = [EncDecRecurrent(n_vis=n_input_at_each_timestep, n_hid=rec_n_hid, return_indices=[-1]), Softmax(n_vis=rec_n_hid, n_classes=n_classes)]
	
	# single layer
	#layers = [EncDecRecurrent(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep, layer_name='recurrent', return_indices=[-1]), Softmax(n_vis=n_input_at_each_timestep, n_classes=n_classes)]

	# double layer
	#layers = [EncDecRecurrent(n_vis=n_input_at_each_timestep, n_hid=rec_n_hid, layer_name='rec_1'), EncDecRecurrent(n_vis=rec_n_hid, n_hid=n_input_at_each_timestep, layer_name='rec_2',return_indices=[-1]), Softmax(n_vis=n_input_at_each_timestep, n_classes=n_classes)]

	# lstm
	#layers = [LSTM(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep, layer_name='lstm', return_indices=[-1], dropout_prob=0.3), Softmax(n_vis=n_input_at_each_timestep, n_classes=n_classes)]

	# 2*lstm
	#layers = [LSTM(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep, layer_name='lstm_1', dropout_prob=0.2),LSTM(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep, layer_name='lstm_2', dropout_prob=0.2, return_indices=[-1]), Softmax(n_vis=n_input_at_each_timestep, n_classes=n_classes)]

	encoding_rec_filepath = '/Users/wulfe/Dropbox/Start/scripts/machine_learning/stacked_enc_dec_rnn/models/enc_dec_overlap_1.save'
	lstm_filepath = '/Users/wulfe/Dropbox/Start/scripts/machine_learning/stacked_enc_dec_rnn/models/lstm_1.save'
	softmax_filepath = '/Users/wulfe/Dropbox/Start/scripts/machine_learning/stacked_enc_dec_rnn/models/softmax_1.save'
	encoding_rec = load_model(encoding_rec_filepath)
	# recurrent_1 = load_model(lstm_filepath)
	# softmax = load_model(softmax_filepath)

	# encoding_rec = encoding_recurrent_overlap.EncodingRecurrentOverlap(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep, layer_name='enc_1')

	# print('building pretrainer...')
	# pre_cost, pre_updates = encoding_rec.get_pretraining_cost_updates(x, learning_rate=0.001)
	
	# pretrainer = theano.function(
	# 	[index],
	# 	[pre_cost],
	# 	updates=pre_updates,
	# 	givens={
	# 		x: trainset_X[index]
	# 	},
	# 	mode='FAST_RUN'
	# )

	# print('pretraining model...')
	# n_epochs = 20
	# n_train_examples = trainset_X.shape.eval()[0]
	# for epoch in range(n_epochs):
	# 	costs = []
	# 	#random_indices = get_random_indices(max_index=n_train_examples - 1, samples_per_epoch=10)
	# 	for sample_idx in range(n_train_examples):
	# 	#for sample_idx in random_indices:
	# 		costs.append(pretrainer(sample_idx)[0])
	# 	print('training cost for epoch {0}: {1}'.format(epoch, np.mean(costs)))

	# for param in encoding_rec.reconstruction_params:
	# 		print('{}: {}'.format(param.name, param.get_value()))

	
	# save_model(encoding_rec, encoding_rec_filepath)


	recurrent_1 = LSTM(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep, layer_name='rec_1', return_indices=[-1], dropout_prob=0.3)
	# #recurrent_2 = LSTM(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep, layer_name='rec_2', return_indices=[-1], dropout_prob=0.2)
	softmax = Softmax(n_vis=n_input_at_each_timestep, n_classes=n_classes)


	# 1*encoding + recurrent
	layers = [encoding_rec, recurrent_1, softmax]
	# layers = [recurrent_1, softmax]

	# 3*lstm
	# layers = [LSTM(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep, layer_name='lstm_1'),
	# 		LSTM(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep, layer_name='lstm_2'),
	# 		LSTM(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep, layer_name='lstm_3', return_indices=[-1]), 
	# 		Softmax(n_vis=n_input_at_each_timestep, n_classes=n_classes)]

	# rnn = EncDecRNN(layers, cost=model_cost, return_indices=[-1])
	cost = Softmax.negative_log_likelihood
	rnn = EncDecRNN(layers, cost=cost, return_indices=[-1])

	# cost, updates = rnn.get_cost_updates((x, print_target))
	cost, updates = rnn.get_cost_updates((x, target))

	print('building trainer...')
	trainer = theano.function(
		[index],
		[cost],
		updates=updates,
		givens={
			x: trainset_X[index],
			target: trainset_y[index]
		},
		mode='FAST_RUN'
	)

	errors = rnn.layers[-1].errors(target)
	validate_model = theano.function(
		inputs=[index],
		outputs=[cost, errors],
		givens={
			x: testset_X[index],
			target: testset_y[index]
		},
		mode='FAST_RUN'
	)

	print('training model...')
	n_train_examples = trainset_X.shape.eval()[0]
	n_test_examples = testset_X.shape.eval()[0]

	n_epochs = 100
	lowest_cost = -1
	for epoch in range(n_epochs):
		costs = []
		#random_indices = get_random_indices(max_index=n_train_examples - 1, samples_per_epoch=100)
		for sample_idx in range(n_train_examples):
		# for sample_idx in random_indices:
			costs.append(trainer(sample_idx)[0])
		avg_cost = np.mean(costs)
		print('training cost for epoch {0}: {1}'.format(epoch, avg_cost))

		if lowest_cost == -1 or avg_cost < lowest_cost * 0.98:
			lowest_cost = avg_cost
			run_validation = True
			save_model(recurrent_1, lstm_filepath)
			save_model(softmax, softmax_filepath)

		predictions = []
		if run_validation:
			print('\nvalidation')
			for sample_idx in range(n_test_examples):
				predictions.append(validate_model(sample_idx)[1])
			accuracy = (1 - np.mean(predictions)) * 100
		 	print('accuracy for epoch {0}: {1}%'.format(epoch, accuracy))
		 	run_validation = False

	# print('finished training, final stats:\nfinal cost: {0}\naccuracy: {1}%'.format(np.mean(costs), accuracy))
	print('finished training, final stats:\nfinal cost: {0}'.format(np.mean(costs)))

	for layer in rnn.layers:
		for param in layer.params:
			print('{}: {}'.format(param.name, param.get_value()))

def main_theano_sign_lang_var_len():
	"""
	:description: this trains a model on the sign language data as well, but accounts for variable length sequences and processes batches.
	"""
	print('loading data...')
	n_input_at_each_timestep = 10
	n_classes = 97	# no base 0 considered, there are just 98 of them. May need to be 97
	
	X, y = sign_lang.load_data_from_aggregate_file()
	X, masks = sign_lang.pad_data_to_max_sample_length(X)
	X = X.astype(theano.config.floatX)
	X = np.swapaxes(X, 0, 1)
	masks = np.swapaxes(masks, 0, 1)

	split_idx = int(.8 * X.shape[1])

	X = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)
	masks = theano.shared(np.asarray(masks, dtype=theano.config.floatX), borrow=True)
	y = theano.shared(y, borrow=True)

	trainset_masks = masks[:, :split_idx, :]
	testset_masks = masks[:, split_idx:, :]
	
	trainset_X, trainset_y = X[:, :split_idx, :], y[:split_idx]
	testset_X, testset_y = X[:, split_idx:, :], y[split_idx:]

	index = T.lscalar()
	x = T.tensor3('x')
	target = T.lvector('target')
	print_x = theano.printing.Print('\nx')(x)
	print_target = theano.printing.Print('target')(target)
	mask = T.tensor3('mask')

	print('building model...')

	lstm_1_filepath = '/Users/wulfe/Dropbox/Start/scripts/machine_learning/stacked_enc_dec_rnn/models/lstm_1.save'
	softmax_filepath = '/Users/wulfe/Dropbox/Start/scripts/machine_learning/stacked_enc_dec_rnn/models/softmax_1.save'
	
	# lstm_1 = load_model(lstm_1_filepath)
	# softmax = load_model(softmax_filepath)


	lstm_1 = variable_length_sequence_lstm.LSTM(n_vis=n_input_at_each_timestep, n_hid=n_input_at_each_timestep, layer_name='rec_1', return_indices=[-1], dropout_prob=0.3)
	softmax = variable_length_sequence_lstm.Softmax(n_vis=n_input_at_each_timestep, n_classes=n_classes)

	layers = [lstm_1, softmax]

	cost_expr = variable_length_sequence_lstm.Softmax.negative_log_likelihood
	rnn = variable_length_sequence_lstm.MLP(layers, cost=cost_expr, return_indices=[-1])

	cost, updates = rnn.get_cost_updates(x, target, mask, learning_rate=0.01)

	batch_size = 50

	print('building trainer...')
	trainer = theano.function(
		[index],
		[cost],
		updates=updates,
		givens={
			x: trainset_X[:, index * batch_size: (index + 1) * batch_size],
			target: trainset_y[index * batch_size: (index + 1) * batch_size],
			mask: trainset_masks[:, index * batch_size: (index + 1) * batch_size]
		},
		mode='FAST_RUN'
	)

	errors = rnn.layers[-1].errors(target)
	validate_model = theano.function(
		inputs=[index],
		outputs=[cost, errors],
		givens={
			x: testset_X[:, index * batch_size: (index + 1) * batch_size],
			target: testset_y[index * batch_size: (index + 1) * batch_size],
			mask: testset_masks[:, index * batch_size: (index + 1) * batch_size]
		},
		mode='FAST_RUN'
	)

	print('training model...')
	n_train_examples = trainset_X.shape.eval()[1]
	n_test_examples = testset_X.shape.eval()[1]

	n_epochs = 100
	lowest_cost = -1
	n_train_batches = int(trainset_X.shape.eval()[1] / float(batch_size))
	n_validation_batches = int(testset_X.shape.eval()[1] / float(batch_size))
	for epoch in range(n_epochs):
		costs = []
		#random_indices = get_random_indices(max_index=n_train_examples - 1, samples_per_epoch=100)
		for sample_idx in range(n_train_batches):
		# for sample_idx in random_indices:
			costs.append(trainer(sample_idx)[0])
		avg_cost = np.mean(costs)
		print('training cost for epoch {0}: {1}'.format(epoch, avg_cost))

		if lowest_cost == -1 or avg_cost < lowest_cost * 0.98:
			lowest_cost = avg_cost
			run_validation = True
			save_model(lstm_1, lstm_1_filepath)
			save_model(softmax, softmax_filepath)

		predictions = []
		if run_validation:
			print('\nvalidation')
			for sample_idx in range(n_validation_batches):
				predictions.append(validate_model(sample_idx)[1])
			accuracy = (1 - np.mean(predictions)) * 100
		 	print('accuracy for epoch {0}: {1}%'.format(epoch, accuracy))
		 	run_validation = False

	# print('finished training, final stats:\nfinal cost: {0}\naccuracy: {1}%'.format(np.mean(costs), accuracy))
	print('finished training, final stats:\nfinal cost: {0}'.format(np.mean(costs)))

	for layer in rnn.layers:
		for param in layer.params:
			print('{}: {}'.format(param.name, param.get_value()))

def main_theano_softmax():
	print('loading data...')
	n_examples = 20
	n_classes = 4
	data_generator = easy_softmax_data_generator
	trainset, testset = get_dataset_toy_raw(data_generator=data_generator,
											n_classes=n_classes, 
											n_examples=n_examples)
	trainset_X, trainset_y = trainset
	testset_X, testset_y = testset
	print('trainset_X.eval(): {}'.format(trainset_X.eval()))
	print('trainset_y.eval(): {}'.format(trainset_y.eval()))

	print('building model...')
	index = T.lscalar()
	x = T.vector('x')
	print_x = theano.printing.Print('\nx')(x)
	target = T.lscalar('target')
	print_target = theano.printing.Print('target')(target)
	softmax = Softmax(n_vis=n_classes+1, n_classes=n_classes)
	cost = softmax.negative_log_likelihood
	rnn = EncDecRNN([softmax], cost=cost, return_indices=[-1])
	cost, updates = rnn.get_cost_updates((print_x, print_target))

	print('building trainer...')
	trainer = theano.function(
		[index],
		[cost],
		updates=updates,
		givens={
			x: trainset_X[index],
			target: trainset_y[index]
		},
		mode='FAST_RUN'
	)

	errors = rnn.layers[-1].errors(print_target)
	validate_model = theano.function(
		inputs=[index],
		outputs=[errors],
		givens={
			x: testset_X[index],
			target: testset_y[index]
		},
		mode='FAST_RUN'
	)

	print('training model...')
	n_epochs = 200
	for epoch in range(n_epochs):
		costs = []
		for sample_idx in range(n_examples):
			costs.append(trainer(sample_idx)[0])

		print('\n\ntraining cost for epoch {0}: {1}\n\n'.format(epoch, np.mean(costs)))

		predictions = []
		if epoch % 5 == 0:
			print('\nvalidation')
			for sample_idx in range(n_examples):
				predictions.append(validate_model(sample_idx))
			accuracy = (1 - np.mean(predictions)) * 100
		 	print('accuracy for epoch {0}: {1}'.format(epoch, accuracy))

	print('finished training, final stats:\nfinal cost: {0}\naccuracy: {1}%'.format(np.mean(costs), accuracy))

	for layer in rnn.layers:
		for param in layer.params:
			print('{}: {}'.format(param.name, param.get_value()))

if __name__ == '__main__':
	main_theano_sign_lang_var_len()


