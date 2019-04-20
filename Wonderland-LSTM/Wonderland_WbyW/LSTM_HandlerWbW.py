# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from string import ascii_lowercase as alphabet
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
# load ascii text and covert to lowercase
filename = "AliceWonderland_Sample.txt"
raw_text = open(filename).read().replace('\n', ' ')
raw_text = raw_text.lower()
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
alpha = set(alphabet)
bad = set(chars) - (set(alpha).union(set(' ')))
for i in bad:
    raw_text = raw_text.replace(i, '')
vocab = list(raw_text.split(" "))
#vocab = filter(lambda a: a != '', vocab)
vocabulary = list(set(vocab) - set(bad).union(set(' ')))
char_to_int = dict((c, i) for i, c in enumerate(vocabulary))
# summarize the loaded data
n_chars = len(vocab)
n_vocab = len(vocabulary)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 10
dataX = []
dataY = []
for i in range(0, len(vocab) - seq_length, 1):
    seq_in = vocab[i:i + seq_length]
    seq_out = vocab[i + seq_length]
    dataX.append([char_to_int[word] for word in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
epochs = 500
# define the LSTM model
def new_model(learning_rate=0.1, momentum=0.8):
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
	model.add(Dropout(0.5))
	model.add(LSTM(256))
	model.add(Dropout(0.3))
	model.add(Dense(y.shape[1], activation='softmax'))
	#decay_rate = learning_rate / epochs
	#Stochastic_Gradient_Descent = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model

def grid_search():
	global epochs
	model = KerasRegressor(build_fn=new_model, verbose=1)
	# the grid_search parameters
	learning_rate = [0.01, 0.1, 0.3]
	momentum = [0.6, 0.8, 0.9]
	batch_size = [32, 64, 128, 256]
	param_grid = dict(learning_rate=learning_rate, momentum=momentum, batch_size=batch_size)
	grid = GridSearchCV(estimator=model, param_grid=param_grid)
	grid_result = grid.fit(X, y, shuffle=False)
	# summary of the results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))

def normal_run():
	global epochs
	model = new_model() #learning_rate=0.3, momentum=0.6)
	# define the checkpoint
	filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	history = list()
	# fit the model
	#for i in range(epochs):
	#    print("Epoch %i/%i:" % (i, epochs))
	#    history.append(model.fit(X, y, epochs=1, batch_size=16, callbacks=callbacks_list, shuffle=False, verbose=1).history['loss'][0])
	#    model.reset_states()
	history = model.fit(X, y, epochs=epochs, batch_size=16, callbacks=callbacks_list)
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['train'], loc='upper left')
	plt.show()

normal_run()
