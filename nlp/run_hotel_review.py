"""
:description: script to train the encoder-decoder rnn
"""
import sys
sys.path.append('/Users/wulfe/Dropbox/Start/scripts/machine_learning')

import hotel_review_data_utils
import sign_lang
import hotel_review_enc_dec_rnn
import variable_length_sequence_lstm
 

def run_hotel_review():
	print('loading data...')
	X = hotel_review_data_utils.load_data_train()
	X_reverse = hotel_review_utils.reverse_X()
	X, masks = sign_lang.pad_data_to_max_sample_length(X)
	X_reverse, _ = sign_lang.pad_data_to_max_sample_length(X_reverse)

	X = X.astype(theano.config.floatX)
	X_reverse = X_reverse.astype(theano.config.floatX)
	masks = masks.astype(theano.config.floatX)

	X = np.swapaxes(X, 0, 1)
	X_reverse = np.swapaxes(X_reverse, 0, 1)
	masks = np.swapaxes(masks, 0, 1)

	X = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)
	masks = theano.shared(np.asarray(masks, dtype=theano.config.floatX), borrow=True)
	X_reverse = theano.shared(np.asarray(X_reverse, dtype=theano.config.floatX), borrow=True)

	index = T.lscalar()
	x = T.tensor3('x')
	target = T.tensor3('target')
	print_x = theano.printing.Print('\nx')(x)
	print_target = theano.printing.Print('target')(target)
	mask = T.tensor3('mask')

	print('building model...')

	# encoder_filepath = '/Users/wulfe/Dropbox/Start/scripts/machine_learning/stacked_enc_dec_rnn/models/enc.save'
	# decoder_filepath = '/Users/wulfe/Dropbox/Start/scripts/machine_learning/stacked_enc_dec_rnn/models/dec.save'
	
	# encoder = load_model(encoder_filepath)
	# decoder = load_model(decoder_filepath)

	# the number of words in the dictionary, including the marker for end-of-document
	n_classes = 25000
	n_hidden = 1000
	encoder = variable_length_sequence_lstm.LSTM(n_vis=n_classes, n_hid=n_hidden, layer_name='enc', return_indices=[-1])
	decoder = hotel_review_enc_dec_rnn.DecoderLSTM(n_hid=n_hidden, n_classes=n_classes, layer_name='dec')

	rnn = hotel_review_enc_dec_rnn.EncoderDecoderRNN(encoder, decoder)

	cost, updates = rnn.get_cost_updates(x, target, mask, learning_rate=0.1)

	batch_size = 10

	print('building trainer...')
	trainer = theano.function(
		[index],
		[cost],
		updates=updates,
		givens={
			x: X_reverse[:, index * batch_size: (index + 1) * batch_size],
			target: X[:, index * batch_size: (index + 1) * batch_size],
			mask: masks[:, index * batch_size: (index + 1) * batch_size]
		},
		mode='FAST_RUN'
	)

	print('training model...')
	n_examples = X.shape.eval()[1]
	n_batches = int(n_examples / float(batch_size))
	n_epochs = 100
	lowest_cost = -1
	for epoch in range(n_epochs):
		costs = []
		for sample_idx in range(n_batches):
			costs.append(trainer(sample_idx)[0])
		avg_cost = np.mean(costs)
		print('training cost for epoch {0}: {1}'.format(epoch, avg_cost))

		if lowest_cost == -1 or avg_cost < lowest_cost * 0.99:
			lowest_cost = avg_cost
			save_model(encoder, encoder_filepath)
			save_model(decoder, decoder_filepath)

	print('finished training, final stats:\nfinal cost: {0}'.format(np.mean(costs)))

	layers = [rnn.encoder, rnn.decoder]
	for layer in layers:
		for param in layer.params:
			print('{}: {}'.format(param.name, param.get_value()))

if __name__ == '__main__':
	run_hotel_review()

