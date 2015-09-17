"""
:description: This file contains 3 classes:
		(1) An encoder-decoder recurrent neural net that contains an encoder and a deocder and orchestrates forward and backward propagation through them
		(2) An encoder, which takes a input sequence and generates a hidden state aka representation of that sequence
		(3) A decoder, which takes a hidden state as input and generates output 
"""

import numpy as np
import theano
from theano import scan
import theano.tensor as T

from pylearn2.expr.nnet import arg_of_softmax
from pylearn2.utils import sharedX


class EncoderDecoderRNN(object):

	def __init__(self, 
				encoder, 
				decoder,
				cost=None):
		"""
		:description: A model that contains an encoder and decoder and orchestrates their combined usage and training

		
		"""
		self.encoder = encoder
		self.decoder = decoder
		self.cost = cost
		self.return_indices = return_indices

	def fprop(self, input, mask):
		return self.decoder.fprop(self.encoder.fprop(input, mask))
		
	def get_cost_updates(self, inputs, targets, mask, learning_rate=0.001, momentum=0.2):
		predictions = self.fprop(inputs, mask)

		if self.cost is not None:
			cost = self.cost(predictions, targets)
		else:
			cost = T.mean(T.sqr(targets - predictions))

		params = self.get_params()

		# this does not work
		try:
			self.gparams = momentum * self.gparams + (1 - momentum) * T.grad(cost, params)
		except:
			self.gparams = T.grad(cost, params)

		updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, self.gparams)]

		return (cost, updates)

	def get_params(self):
		return self.encoder.params + self.decoder.params

class DecoderLSTM(object):
	"""
	:description: A decoder class. Takes a hidden state and generates an output sequence.
	"""
	def __init__(self, 
				n_hid, 
				n_classes, 
				layer_name, 
				rng=None,
				return_indices=None,
				param_init_range=0.02,
				forget_gate_init_bias=0.05,
				input_gate_init_bias=0.,
				output_gate_init_bias=0.,
				dropout_prob=0.0):

		self.n_hid = n_hid
		self.n_classes = n_classes
		self.layer_name = layer_name
		self.param_init_range = param_init_range
		self.return_indices = return_indices
		self.forget_gate_init_bias = forget_gate_init_bias
		self.input_gate_init_bias = input_gate_init_bias
		self.output_gate_init_bias = output_gate_init_bias

		# only create random arrays once and reuse via copy()
		irange = self.param_init_range
		# input-to-hidden array, used for incorporating the generated output (conditioned on output)
		init_Wxh = self.rng.uniform(-irange, irange, (self.n_classes, self.n_hid))
		# hidden-to-hidden array
		init_Whh = self.rng.uniform(-irange, irange, (self.n_hid, self.n_hid))
		# hidden-to-output array, used only by the 'softmax' portion of the decoder
		init_Whx = self.rng.uniform(-irange, irange, (self.n_hid, self.n_classes))

		# input-to-hidden array, used for incorporating the generated output 
		self.Wxh = theano.shared(value=init_Wxh, name=self.layer_name + '_Wxh', borrow=True)
		self.bxh = theano.shared(value=np.zeros(self.n_hid), name='bhx', borrow=True)
		# hidden-to-hidden (rows, cols) = (n_hidden, n_hidden)
		self.Whh = theano.shared(value=init_Whh, name=self.layer_name + '_Whh', borrow=True)
		# hidden-to-output (rows, cols) = (n_hidden, n_classes)
		self.Whx = theano.shared(value=init_Whx, name=self.layer_name + '_Whx', borrow=True)
		self.bhx = theano.shared(value=np.zeros(self.n_classes), name='bhx', borrow=True)

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

		self.params = [self.Wxh, self.bxh, self.Whh, self.Whx, self.bhx, self.O_b, self.O_x, self.O_h, self.O_c, self.I_b, self.I_x, self.I_h, self.I_c, self.F_b, self.F_x, self.F_h, self.F_c]


	def fprop(self, encoding):
		"""
		:description: calls decode function. Just here for some consistency.
		"""
		return self.decode(encoding)

	def decode(self, encoding):
		"""
		:description: decodes an encoding into an output sequence.

		:type encoding: tensor3
		:param encoding: a batch of encodings with the shape (n_time_steps, n_batches, n_hidden). The reason n_time_steps takes the first dimension spot is that this allows for processing with the theano.scan function.
		"""
		pass

	def decode_step(self, ):
		pass



