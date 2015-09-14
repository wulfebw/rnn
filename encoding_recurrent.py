"""
:description: Second version of the encoding recurrent layer. Want to keep it separate so different file.

:development:

	:todo:
		(1) unit test EncodingRecurrent completely

	:to try:
		(1) remove encoding layer params from the RNN params so that they aren't updated
		(2) variable length sequences


"""

import numpy as np
import theano
from theano import scan
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

class EncodingRecurrent(object):

	def __init__(self, n_vis, n_hid, layer_name, rng=None, encoding_length=4, offset_len=1, param_init_range=0.02):

		self.n_vis = n_vis
		self.n_hid = n_hid
		self.layer_name = layer_name
		assert encoding_length > offset_len, 'encoding_length must be greater than offset_len'
		self.encoding_length = encoding_length
		self.offset_len = offset_len
		self.nonlinearity = T.tanh
		if rng is None:
			rng = np.random.RandomState()
		self.rng = rng
		self.theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))
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
		#self.params = []

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

		total_sequence_length = T.cast(state_below.shape[0], theano.config.floatX)
		self.n_encodings = T.cast(T.ceil(total_sequence_length / self.encoding_length), 'int32')
		self.n_padding_timesteps = T.cast(self.n_encodings * self.encoding_length - total_sequence_length, 'int32')
		zeros = T.alloc(np.cast[theano.config.floatX](0), self.n_padding_timesteps, self.n_vis)
		state_below = T.concatenate((zeros, state_below))

		Wxh = self.Wxh
		bxh = self.bxh
		Whhe = self.Whhe

		state_below = state_below.reshape((self.encoding_length, self.n_encodings, self.n_vis))
		state_below = T.dot(state_below, Wxh) + bxh
		
		# a single output will be n_encoding rows with n_hid features each
		encoding_0 = T.alloc(np.cast[theano.config.floatX](0), self.n_encodings, self.n_hid)

		encodings, updates = scan(fn=self.encode_step, sequences=[state_below], outputs_info=[encoding_0], non_sequences=[Whhe])
		# encodings is a 3d vector (encoding_length, n_encodings, n_hid)
		# returns encodings[-1] in 2d vector shape = (n_encodings, n_hid)
		return encodings[-1]

	def encode_step(self, state_below, state_before, Whhe):
		return self.nonlinearity(T.dot(state_before, Whhe) + state_below)

	def decode(self, encodings):
		"""
		:type encodings: theano matrix (2d tensor)
		:param encodings: the matrix of final hidden states, where each row corresponds to a hidden state (and each col corresponds to a feature/value of a node)
		"""
		Whhd = self.Whhd
		Wxh = self.Wxh
		Whx = self.Whx
		bhx = self.bhx

		reconstructed_input_0 = T.alloc(np.cast[theano.config.floatX](0), self.n_encodings, self.n_vis)

		([reconstructed_hidden_states, reconstructed_input], updates) = scan(fn=self.decode_step, outputs_info=[encodings, reconstructed_input_0], non_sequences=[Whhd, Wxh, Whx, bhx], n_steps=self.encoding_length)

		reconstructed_input = reconstructed_input.reshape((self.encoding_length * self.n_encodings, self.n_vis))
		reconstructed_input = self.truncate_to_original_sequence_length(reconstructed_input)
		return reconstructed_input
		
	def decode_step(self, prev_hidden_state, prev_reconstructed_input, Whhd, Wxh, Whx, bhx):
		cur_hidden_state = self.nonlinearity(T.dot(prev_hidden_state, Whhd) + T.dot(prev_reconstructed_input, Wxh))
		cur_reconstructed_input = self.nonlinearity(T.dot(cur_hidden_state, Whx) + bhx)
		return [cur_hidden_state, cur_reconstructed_input]

	def get_corrupted_input_sequence(self, input_sequence, corruption_level=0.0):
		mask = self.theano_rng.binomial(size=input_sequence.shape, n=1, p=1 - corruption_level, dtype=theano.config.floatX) 
		return mask * input_sequence

	def truncate_to_original_sequence_length(self, input_sequence):
		return input_sequence[self.n_padding_timesteps:]

	def get_pretraining_cost_updates(self, input_sequence, learning_rate=0.005):
		"""
		:description: reconstruction cost

		"""
		corrupted_input_sequence = self.get_corrupted_input_sequence(input_sequence, corruption_level=0.25)
		reconstructed_input_sequence = self.decode(self.encode(corrupted_input_sequence))
		cost = T.sum(T.sqr(input_sequence - reconstructed_input_sequence))

		gparams = T.grad(cost, self.reconstruction_params)
		updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.reconstruction_params, gparams)]

		return (cost, updates)


