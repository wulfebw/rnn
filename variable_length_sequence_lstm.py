"""
:description: incorporate variable length sequence functionality
"""

import numpy as np
import theano
from theano import scan
import theano.tensor as T

from pylearn2.expr.nnet import arg_of_softmax
from pylearn2.utils import sharedX


class MLP(object):

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

	def fprop(self, input, mask):
		state_below = input
		outputs = []
		for layer in self.layers:
			state_below = layer.fprop(state_below, mask)
			outputs.append(state_below)

		if self.return_indices is not None:
			if len(self.return_indices) > 1:
				return [outputs[idx] for idx in self.return_indices]
			else:
				return outputs[self.return_indices[0]]
		else:
			return outputs

	def get_cost_updates(self, input, targets, mask, learning_rate=0.01):
		predictions = self.fprop(input, mask)

		if self.cost is not None:
			cost = self.cost(predictions, targets)
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

	def fprop(self, state_below, mask=None):
		"""
		:development: 
			(1) what is the shape of state_below? Does it account for batches?
				- let's assume that it uses the (time, batch, data) approach in the original code, so need some changes
			(2) do _scan_updates do anything important?

		"""

		#z0 = T.alloc(np.cast[theano.config.floatX](0), self.n_hid)
		#c0 = T.alloc(np.cast[theano.config.floatX](0), self.n_hid)
		# return a (batch_size, n_hid) vector, state_below.shape[1] == batch_size
		z0 = T.alloc(np.cast[theano.config.floatX](0), state_below.shape[1], self.n_hid)
		c0 = T.alloc(np.cast[theano.config.floatX](0), state_below.shape[1], self.n_hid)

		if state_below.shape[1] == 1:
			z0 = T.unbroadcast(z0, 0)
			c0 = T.unbroadcast(c0, 0)

		Wxh = self.Wxh
		Whh = self.Whh
		bxh = self.bxh
		state_below_input = T.dot(state_below, self.I_x) + self.I_b
		state_below_forget = T.dot(state_below, self.F_x) + self.F_b
		state_below_output = T.dot(state_below, self.O_x) + self.O_b
		state_below = T.dot(state_below, Wxh) + bxh

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
			z = z * mask + (1 - mask) * state_before
			return z, c_t
			#return z.astype(theano.config.floatX), c_t.astype(theano.config.floatX)

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

	def fprop(self, state_below, mask=None):

		prob_y_given_x = T.nnet.softmax(T.dot(state_below, self.W) + self.b)
		print_prob_y_given_x = theano.printing.Print('prob_y_given_x')(prob_y_given_x)
		# T.argmax returns the index of the greatest element in the vector prob_y_given_x 
		self.y_pred = T.argmax(print_prob_y_given_x, axis=1)
		self.y_pred_print = theano.printing.Print('y_pred')(self.y_pred)
		# return print_prob_y_given_x
		return prob_y_given_x

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError("""y should have the same number of dimensions as y_pred, but y had the dim: {0} and y_pred had dim: {1}""".format(y.ndim, self.y_pred.ndim))

		# if error == 1, then y_pred != y, if error == 0 then y_pred == y
		error = T.mean(T.neq(self.y_pred_print, y))
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
		return -T.mean(T.log(prob_y_given_x)[T.arange(target.shape[0]), target])
		#return -T.log(prob_y_given_x[target])


