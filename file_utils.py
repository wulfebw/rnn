import cPickle

def save_model(model, output_filename):
	f = file(output_filename, 'wb')
	cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()

def load_model(model_filename):
	f = file(model_filename, 'rb')
	model = cPickle.load(f)
	f.close()
	return model