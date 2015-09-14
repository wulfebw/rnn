"""
:description:

	:model concepts:
		(1) the options for different output formats
			- the entire sequence 
				- only the hidden values
				- the hidden values transformed by an output matrix
				- the last hidden value
				- the last hidden value transformed by an output matrix
		(2) the options for stacking these
			- if you pass the hidden states to the next layer, then you have a stacked rnn
			- if you pass an encoding you have an stacked enc dec
:development:

	:left off: 
		(1) the sub in the pretraining costs is not subbing same lenth sequences (i.e., the reconstructed input is a different length than the original input, that doesn't make sense?)

	:current goal:
		(1) make encoding layer work 

	:plan:
		(1) make encoding layer work faster
		(2) try to prove to some extent that the endoing layer 'works'
		(3) make changes to encoding layer to make it work
			(a) mean instead of max
			(b) offset increase
			(c) unit test the merging method
				(i) max merge 
				(ii) mean merge
			(d) try pretraining

	:issues:
		(1) encoding layer might not work
		(2) even if encoding layer works it is really slow, impractically slow
		(3) need to use mask for different length sequences in the same dataset?

	:todo:
		(1) 
		(2) different length sequences
		(3) try out sign_lang on gpu 
		(4) corrupted input for EncodingRecurrent
		(5) try different training algo
		(6) incorp dropout
		(7) incorp corruption

"""

import numpy as np
import theano
from theano import scan
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from pylearn2.expr.nnet import arg_of_softmax
from pylearn2.utils import sharedX



class EncDecRNN(object):

	def __init__(self, 
				layers, 
				cost=None,
				return_indices=None):
		"""
		:description:

		:type return_indices: list of ints
		:param return_indices: specifies which layer-outputs should be returned. return_indices = [-1] returns the output from only the final layer.
		"""
		self.layers = layers
		self.cost = cost
		self.return_indices = return_indices

	def fprop(self, input):
		state_below = input
		outputs = []
		for layer in self.layers:
			state_below = layer.fprop(state_below)
			outputs.append(state_below)
			# outputs.append(layer.fprop(state_below))
			# state_below = layer.encode(state_below)

		if self.return_indices is not None:
			if len(self.return_indices) > 1:
				return [outputs[idx] for idx in self.return_indices]
			else:
				return outputs[self.return_indices[0]]
		else:
			return outputs

	def get_pretraining_cost_updates(self):
		pass

	def get_cost_updates(self, data, learning_rate=0.01):
		input, target = data
		predictions = self.fprop(input)

		if self.cost is not None:
			cost = self.cost(predictions, target)
		else:
			cost = T.mean(T.sqr(targets - predictions))

		params = self.get_fprop_params()
		gparams = T.grad(cost, params)
		updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]

		return (cost, updates)

	def get_fprop_params(self):
		params = []
		for layer in self.layers:
			params += layer.params
		return params

class EncDecRecurrent(object):

	def __init__(self,
				n_vis,
				n_hid,
				layer_name,
				rng=None,
				return_indices=None,
				param_init_range=0.02,
				dropout_prob=0.0
				):
		"""
		:description:

		:type return_indices: list of ints
		:param return_indices: specifies which timestep outputs should be returned from this layer. return_indices = [-1] returns only the final timestep output
		"""		
		if rng is None:
			rng = np.random.RandomState()
		self.rng = rng
		self.n_vis = n_vis
		self.n_hid = n_hid
		self.layer_name = layer_name
		self.param_init_range = param_init_range
		self.return_indices = return_indices
		self.dropout_prob = dropout_prob

		# input-to-hidden (rows, cols) = (n_visible, n_hidden)
		init_Wxh = rng.uniform(-self.param_init_range, self.param_init_range, (self.n_vis, self.n_hid))
		self.Wxh = theano.shared(value=init_Wxh, name=self.layer_name + '_Wxh', borrow=True)
		self.bxh = theano.shared(value=np.zeros(self.n_hid), name=self.layer_name + '_bxh', borrow=True)
		# hidden-to-hidden (rows, cols) = (n_hidden, n_hidden) for both encoding and decoding ('tied weights')
		init_Whh = rng.uniform(-self.param_init_range, self.param_init_range, (self.n_hid, self.n_hid))
		self.Whh = theano.shared(value=init_Whh, name=self.layer_name + '_Whh', borrow=True)
		self.bhh = theano.shared(value=np.zeros(self.n_hid), name=self.layer_name + '_bhh', borrow=True)
		# hidden-to-output matrix (rows, cols) = (n_hidden, n_visible)
		init_Who = rng.uniform(-self.param_init_range, self.param_init_range, (self.n_hid, self.n_vis))
		self.Who = theano.shared(value=init_Who, name=self.layer_name + '_Who', borrow=True)
		self.bho = theano.shared(value=np.zeros(self.n_vis), name=self.layer_name + '_bho', borrow=True)

		# reconstruct input
		# self.params = [self.Wxh, self.bxh, self.Whh, self.bhh, self.Who, self.bho]

		self.params = [self.Wxh, self.bxh, self.Whh, self.bhh]

		self.nonlinearity = T.tanh

	def fprop(self, state_below):
		"""
		:description:

		:type state_below: theano matrix
		:param state_below: a two dimensional matrix where the first dim represents time and the second dim represents features: shape = (time, features)
		"""

		#init_output = T.alloc(np.cast[theano.config.floatX](0), state_below.shape[0], self.n_hid)
		init_output = T.alloc(np.cast[theano.config.floatX](0), self.n_hid)
		Wxh, bxh, Whh, bhh, Who, bho = self.Wxh, self.bxh, self.Whh, self.bhh, self.Who, self.bho
		state_below = T.dot(state_below, Wxh) + bxh

		if state_below.shape[0] == 1:
			init_output = T.unbroadcast(init_output, 0)
		if self.n_hid == 1:
			init_output = T.unbroadcast(init_output, 1)

		def fprop_step(state_below_timestep, state_before_timestep, Whh, bhh):
			return self.nonlinearity(state_below_timestep + T.dot(state_before_timestep, Whh) + bhh)

		outputs, updates = scan(fn=fprop_step, sequences=[state_below], outputs_info=[init_output], non_sequences=[Whh, bhh])

		# reconstruct input
		# outputs = T.dot(outputs, Who) + bho

		if self.return_indices is not None:
			if len(self.return_indices) > 1:
				return [outputs[idx] for idx in self.return_indices]
			else:
				return outputs[self.return_indices[0]]
		else:
			return outputs

class LSTM(object):

	def __init__(self,
				n_vis,
				n_hid,
				layer_name,
				rng=None,
				return_indices=None,
				param_init_range=0.02,
				forget_gate_init_bias=0.05,
				input_gate_init_bias=0.,
				output_gate_init_bias=0.,
				dropout_prob=0.0
				):
		if rng is None:
			rng = np.random.RandomState()
		self.rng = rng
		self.n_vis = n_vis
		self.n_hid = n_hid
		self.layer_name = layer_name
		self.param_init_range = param_init_range
		self.return_indices = return_indices
		self.forget_gate_init_bias = forget_gate_init_bias
		self.input_gate_init_bias = input_gate_init_bias
		self.output_gate_init_bias = output_gate_init_bias
		self.dropout_prob = dropout_prob

		# only create random arrays once and reuse via copy()
		irange = self.param_init_range
		init_Wxh = self.rng.uniform(-irange, irange, (self.n_vis, self.n_hid))
		init_Whh = self.rng.uniform(-irange, irange, (self.n_hid, self.n_hid))

		# input-to-hidden (rows, cols) = (n_visible, n_hidden)
		self.Wxh = theano.shared(value=init_Wxh, name=self.layer_name + '_Wxh', borrow=True)
		self.bxh = theano.shared(value=np.zeros(self.n_hid), name='bxh', borrow=True)
		# hidden-to-hidden (rows, cols) = (n_hidden, n_hidden) for both encoding and decoding ('tied weights')
		self.Whh = theano.shared(value=init_Whh, name=self.layer_name + '_Whh', borrow=True)

		# lstm parameters
		# Output gate switch
		self.O_b = sharedX(np.zeros((self.n_hid,)) + self.output_gate_init_bias, name=(self.layer_name + '_O_b'))
		self.O_x = sharedX(init_Wxh, name=(self.layer_name + '_O_x'))
		self.O_h = sharedX(init_Whh, name=(self.layer_name + '_O_h'))
		self.O_c = sharedX(init_Whh.copy(), name=(self.layer_name + '_O_c'))
		# Input gate switch
		self.I_b = sharedX(np.zeros((self.n_hid,)) + self.input_gate_init_bias, name=(self.layer_name + '_I_b'))
		self.I_x = sharedX(init_Wxh.copy(), name=(self.layer_name + '_I_x'))
		self.I_h = sharedX(init_Whh.copy(), name=(self.layer_name + '_I_h'))
		self.I_c = sharedX(init_Whh.copy(), name=(self.layer_name + '_I_c'))
		# Forget gate switch
		self.F_b = sharedX(np.zeros((self.n_hid,)) + self.forget_gate_init_bias, name=(self.layer_name + '_F_b'))
		self.F_x = sharedX(init_Wxh.copy(), name=(self.layer_name + '_F_x'))
		self.F_h = sharedX(init_Whh.copy(), name=(self.layer_name + '_F_h'))
		self.F_c = sharedX(init_Whh.copy(), name=(self.layer_name + '_F_c'))

		self.params = [self.Wxh, self.bxh, self.Whh, self.O_b, self.O_x, self.O_h, self.O_c, self.I_b, self.I_x, self.I_h, self.I_c, self.F_b, self.F_x, self.F_h, self.F_c]

	def fprop(self, state_below):
		"""
		:development: 
			(1) what is the shape of state_below? Does it account for batches?
				- let's assume that it uses the (time, batch, data) approach in the original code, so need some changes
			(2) do _scan_updates do anything important?

		"""

		z0 = T.alloc(np.cast[theano.config.floatX](0), self.n_hid)
		c0 = T.alloc(np.cast[theano.config.floatX](0), self.n_hid)
		# z0 = T.alloc(np.cast[theano.config.floatX](0), state_below.shape[0], self.n_hid)
		# c0 = T.alloc(np.cast[theano.config.floatX](0), state_below.shape[0], self.n_hid)

		if state_below.shape[0] == 1:
			z0 = T.unbroadcast(z0, 0)
			c0 = T.unbroadcast(c0, 0)

		Wxh = self.Wxh
		Whh = self.Whh
		bxh = self.bxh
		state_below_input = T.dot(state_below, self.I_x) + self.I_b
		state_below_forget = T.dot(state_below, self.F_x) + self.F_b
		state_below_output = T.dot(state_below, self.O_x) + self.O_b
		state_below = T.dot(state_below, Wxh) + bxh

		# probability that a given connection is dropped is self.dropout_prob
		# the 'p' parameter to binomial determines the likelihood of returning a 1
		# is the mask value is a 1, then the connection is not dropped
		# therefore 1 - dropout_prob gives the prob of droping a node (aka prob of 0)
		theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))
		mask = theano_rng.binomial(p=self.dropout_prob, size=state_below.shape, dtype=state_below.dtype)

		def fprop_step(state_below, 
						state_below_input, 
						state_below_forget, 
						state_below_output, 
						mask,
						state_before, 
						cell_before, 
						Whh):

			i_on = T.nnet.sigmoid(
				state_below_input +
				T.dot(state_before, self.I_h) +
				T.dot(cell_before, self.I_c)
			)

			f_on = T.nnet.sigmoid(
				state_below_forget +
				T.dot(state_before, self.F_h) +
				T.dot(cell_before, self.F_c)
			)

			c_t = state_below + T.dot(state_before, Whh)
			c_t = f_on * cell_before + i_on * T.tanh(c_t)

			o_on = T.nnet.sigmoid(
				state_below_output +
				T.dot(state_before, self.O_h) +
				T.dot(c_t, self.O_c)
			)
			z = o_on * T.tanh(c_t)

			# either carry the new values (z) or carry the old values (state_before)
			z = z * mask + (1 - mask) * state_before

			return z, c_t

		((z, c), updates) = scan(fn=fprop_step,
								sequences=[state_below,
											state_below_input,
											state_below_forget,
											state_below_output,
											mask],
								outputs_info=[z0, c0],
								non_sequences=[Whh])

		if self.return_indices is not None:
			if len(self.return_indices) > 1:
				return [z[i] for i in self.return_indices]
			else:
				return z[self.return_indices[0]]
		else:
			return z

class EncodingRecurrent(object):

	def __init__(self, n_hid, n_vis, layer_name, rng=None, encoding_length=8, offset_len=1, param_init_range=0.02):

		self.n_hid = n_hid
		self.n_vis = n_vis
		self.layer_name = layer_name
		assert encoding_length > offset_len, 'encoding_length must be greater than offset_len'
		self.encoding_length = encoding_length
		self.offset_len = offset_len
		self.nonlinearity = T.tanh
		if rng is None:
			rng = np.random.RandomState()
		self.rng = rng
		self.param_init_range = param_init_range
		
		# encoding parameters
		init_Wxh = self.rng.uniform(-self.param_init_range, self.param_init_range, (self.n_vis, self.n_hid))
		self.Wxh = theano.shared(value=init_Wxh, name=self.layer_name + '_Wxh', borrow=True)
		self.bxh = theano.shared(value=np.zeros(self.n_hid), name=self.layer_name + '_bxh', borrow=True)
		init_Whhe = self.rng.uniform(-self.param_init_range, self.param_init_range, (self.n_hid, self.n_hid))
		self.Whhe = theano.shared(value=init_Whhe, name=self.layer_name + '_Whhe', borrow=True)
		# decoding parameters, tied weights
		self.Whhd = self.Whhe.T
		self.Whx = self.Wxh.T
		self.bhx = theano.shared(value=np.zeros(self.n_vis), name=self.layer_name + '_bhx', borrow=True)
		# don't include decoding params since weights are tied (except for bhx, still need that)
		self.reconstruction_params = [self.Wxh, self.bxh, self.Whhe, self.bhx]
		# params, called by the containing class to get the gradient, should only include params involved in encoding
		self.params = [self.Wxh, self.bxh, self.Whhe]

	def fprop(self, input):
		"""
		:description: returns an encoding of the input

		:type rval: 2d tensor
		:param rval: a sequence of states that represent an encoding of the original sequence
		"""
		return self.encode(input)

	def encode(self, state_below):
		"""
		:development:
			(1) may need to prepend encoding_length * padding array to the state_below to produce the same length sequence as state_below
			(2) can return an offset encoding by only returing certain indices of the encoding (though this is pretty wasteful)

		:type state_below: 2d tensor
		:param state_below: the enitre sequence of states from the layer below the current one

		:type rval: 2d tensor
		:param rval: an encoding of the state_below (the entire sequence of state) to be passed to the above layer
		"""
		# to make the encodings start with the first state in state_below, prepend encoding_length vectors of value zero
		zeros = T.alloc(np.cast[theano.config.floatX](0), self.encoding_length - 1, self.n_hid)
		state_below = T.concatenate((zeros, state_below))

		encoding_0 = T.alloc(np.cast[theano.config.floatX](0), self.n_hid)
		# negative, reverse indicies for the taps 
		# e.g., [-4, -3, -2, -1, -0] would pass those indicies from state_below to the encode_step
		taps = [-1 * tap for tap in range(self.encoding_length)[::-1]]
		encodings, updates = scan(fn=self.encode_subsequence, sequences=dict(input=state_below, taps=taps), outputs_info=[encoding_0])

		return encodings

	def encode_subsequence(self, *args):
		"""
		:development: the state_below_subseq consists of all the args except the last

		"""
		Wxh = self.Wxh
		bxh = self.bxh
		Whhe = self.Whhe
		state_below_subsequence = list(args[:-1])
		state_below_subsequence = T.dot(state_below_subsequence, Wxh) + bxh

		encoding_0 = T.alloc(np.cast[theano.config.floatX](0), self.n_hid)

		subsequence_encoding, updates = scan(fn=self.encode_subsequence_step, sequences=[state_below_subsequence], outputs_info=[encoding_0], non_sequences=[Whhe])

		return subsequence_encoding[-1]

	def encode_subsequence_step(self, state_below_timestep, state_before_timestep, Whhe):
		return self.nonlinearity(state_below_timestep + T.dot(state_before_timestep, Whhe))

		
	def decode_encodings(self, encodings):
		"""

		:type encoding: 3d tensor
		:param encoding: an encoding of the state_below

		:type rval: 3d tensor
		:param rval: a reconstruction of the original state_below
		"""
		reconstructed_subsequences_0 = T.alloc(np.cast[theano.config.floatX](0), self.encoding_length, self.n_vis)
		reconstructed_subsequences, updates  = scan(fn=self.decode_encoding, sequences=[encodings], outputs_info=[reconstructed_subsequences_0])

		reconstructed_input = self.merge_reconstructed_subsequences(reconstructed_subsequences)
		return reconstructed_input

	def decode_encoding(self, encoding, prev_reconstructed_subsequence):
		"""
		:development: 
			(1) n_steps might need to equal self.encoding_length + or - 1, not sure
		"""
		Whhd = self.Whhd
		Wxh = self.Wxh
		Whx = self.Whx
		bhx = self.bhx

		reconstructed_input_0 = T.alloc(np.cast[theano.config.floatX](0), self.n_vis)

		([reconstructed_hidden_states, reconstructed_inputs], updates) = scan(fn=self.decode_encoding_step, outputs_info=[encoding, reconstructed_input_0], non_sequences=[Whhd, Wxh, Whx, bhx], n_steps=self.encoding_length)

		return reconstructed_inputs

	def decode_encoding_step(self, prev_hidden_state, prev_reconstructed_input, Whhd, Wxh, Whx, bhx):
		cur_hidden_state = self.nonlinearity(T.dot(prev_hidden_state, Whhd) + T.dot(prev_reconstructed_input, Wxh))
		cur_reconstructed_input = self.nonlinearity(T.dot(cur_hidden_state, Whx) + bhx)
		return [cur_hidden_state, cur_reconstructed_input]

	def merge_reconstructed_subsequences(self, subsequences, offset_len=1):
		return subsequences[:,0,:]

		# merged_subsequences = subsequences[0]
		# subsequence_len = len(subsequences[0])
		# for idx, subsequence in enumerate(subsequences[1:]):
		# 	cur_offset = (idx + 1) * offset_len

		# 	# print('cur_offset: {}'.format(cur_offset))
		# 	# print('merged_subsequences[cur_offset:]: {}'.format(merged_subsequences[cur_offset:]))
		# 	# print('subsequence[:-offset_len]: {}\n'.format(subsequence[:-offset_len]))

		# 	# cur_overlap = np.max((merged_subsequences[cur_offset:], subsequence[:-offset_len]), axis=0)
		# 	cur_overlap = T.max((merged_subsequences[cur_offset:], subsequence[:-offset_len]), axis=0)
		# 	merged_subsequences[cur_offset:] = cur_overlap
		# 	# print('cur_overlap: {}'.format(cur_overlap))
		# 	# print('merged_subsequences: {}\n'.format(merged_subsequences))

		# 	cur_addition = subsequence[-offset_len:]
		# 	merged_subsequences += cur_addition
		# 	# print('cur_addition: {}'.format(cur_addition))
		# 	# print('merged_subsequences: {}\n'.format(merged_subsequences))

		# return merged_subsequences

	def get_corrupted_input_sequence(self, input_sequence):
		return input_sequence

	def apply_offset(self, sequence):
		pass

	def get_pretraining_cost_updates(self, input_sequence, learning_rate=0.005):
		"""
		:description: reconstruction cost

		"""
		corrupted_input_sequence = self.get_corrupted_input_sequence(input_sequence)
		reconstructed_input_sequence = self.decode_encodings(self.encode(corrupted_input_sequence))
		cost = T.sum(T.sqr(input_sequence - reconstructed_input_sequence))

		gparams = T.grad(cost, self.reconstruction_params)
		updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.reconstruction_params, gparams)]

		return (cost, updates)


class Softmax(object):

	def __init__(self, n_vis, n_classes, rng=None, param_init_range=0.02):
		"""
		:description: single-batch softmax layer used with recurrent layers. Notice that X and b are of size (n_vis, n_classes + 1) and (n_classes + 1) respectively. This is b/c I want to be able to index into the vector of probabilities using the target value itself (see negative_log_likelihood method).
		"""

		if rng is None:
			rng = np.random.RandomState()

		self.n_vis = n_vis
		self.n_classes = n_classes
		self.param_init_range = param_init_range

		init_W = rng.uniform(-self.param_init_range, self.param_init_range, (self.n_vis, self.n_classes+1))
		self.W = theano.shared(value=init_W, name='W', borrow=True)
		self.b = theano.shared(value=np.zeros(self.n_classes+1), name='b', borrow=True)

		self.params = [self.W, self.b]

	def fprop(self, state_below):
		# prob_y_given_x is a 2d array with only one element e.g., [[1,2,3]], so take only the first ele
		prob_y_given_x = T.nnet.softmax(T.dot(state_below, self.W) + self.b)[0]
		print_prob_y_given_x = theano.printing.Print('prob_y_given_x')(prob_y_given_x)
		# T.argmax returns the index of the greatest element in the vector prob_y_given_x 
		self.y_pred = T.argmax(print_prob_y_given_x)
		self.y_pred_print = theano.printing.Print('y_pred')(self.y_pred)
		# return print_prob_y_given_x
		return prob_y_given_x

	def encode(self, state_below):
		return state_below

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError("""y should have the same number of dimensions as y_pred, but y had the dim: {0} and y_pred had dim: {1}""".format(y.ndim, self.y_pred.ndim))

		# if error == 1, then y_pred != y, if error == 0 then y_pred == y
		error = T.neq(self.y_pred_print, y)
		print_error = theano.printing.Print('error')(error)
		return print_error


	@staticmethod
	def negative_log_likelihood(prob_y_given_x, target):
		"""  
		:description: the negative_log_likelihood over a single example.

		:type prob_y_given_x: vector
		:param prob_y_given_x: vector of probabilities for a single example. 
							Shape of prob_y_given_x is (1, n_classes)

		:type target: int
		:param target: the correct label for this example
		"""
		return -T.log(prob_y_given_x[target])











