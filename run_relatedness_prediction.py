from tqdm import tqdm
import pickle
import numpy as np
import karantools as kt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

GLOVE_RELATEDNESS_PATH = 'outputs/relatedness_glove_data.pkl'
ENRICHED_RELATEDNESS_PATH = 'outputs/relatedness_enriched_data.pkl'

def evaluate_relatedness(train_X, train_y, test_X, test_y):
	model = MLPClassifier(early_stopping=True, verbose=True)
	model.fit(train_X, train_y)

	train_y_pred = model.predict(train_X)
	test_y_pred = model.predict(test_X)

	print('Train accuracy', accuracy_score(train_y, train_y_pred))

	print('Train Confusion Matrix\n', confusion_matrix(train_y, train_y_pred))

	print('Test accuracy', accuracy_score(test_y, test_y_pred))

	print('Test Confusion Matrix\n', confusion_matrix(test_y, test_y_pred))

def main():
	with open(GLOVE_RELATEDNESS_PATH, 'rb') as f:
		glove_X, glove_y = pickle.load(f)

	with open(ENRICHED_RELATEDNESS_PATH, 'rb') as f:
		enriched_X, enriched_y = pickle.load(f)

	kt.assert_eq(glove_X.shape, enriched_X.shape)

	glove_permutation = np.random.permutation(glove_X.shape[0])
	glove_X = glove_X[glove_permutation]
	glove_y = glove_y[glove_permutation]

	enriched_permutation = np.random.permutation(enriched_X.shape[0])
	enriched_X = enriched_X[enriched_permutation]
	enriched_y = enriched_y[enriched_permutation]

	split_idx = int(enriched_X.shape[0] * .9)

	train_glove_X = glove_X[:split_idx]
	test_glove_X = glove_X[split_idx:]

	train_glove_y = glove_y[:split_idx]
	test_glove_y = glove_y[split_idx:]

	train_enriched_X = enriched_X[:split_idx]
	test_enriched_X = enriched_X[split_idx:]

	train_enriched_y = enriched_y[:split_idx]
	test_enriched_y = enriched_y[split_idx:]

	print('Running GloVe\n')
	evaluate_relatedness(train_glove_X, train_glove_y, test_glove_X, test_glove_y)
	print('\nRunning Enriched\n')
	evaluate_relatedness(train_enriched_X, train_enriched_y, test_enriched_X, test_enriched_y)


if __name__=='__main__':
	main()