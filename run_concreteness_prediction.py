import pickle
from torchtext.vocab import GloVe
from tqdm import tqdm
import argparse
import karantools as kt
import utils
import random
random.seed(42)
import functools
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def main(args):
	with open('evaluation/concreteness/concreteness_data.pkl', 'rb') as f:
		ratings, concreteness_stoi, abstract_words, concrete_words = pickle.load(f)

	word_stoi, word_vectors = kt.lazy_load(functools.partial(utils.txt_to_stoi_vectors, args.filename), args.pkl_path)

	word_vector_words = set(word_stoi.keys())
	concreteness_words = set(concreteness_stoi.keys())

	# abstract_word_intersection = word_vector_words.intersection(abstract_words)
	# concrete_word_intersection = word_vector_words.intersection(concrete_words)

	# print('Abstract words', len(abstract_words))
	# print('Concrete words', len(concrete_words))

	# print('Abstract word intersection with word vectors', len(abstract_word_intersection))
	# print('Concrete word intersection with word vectors', len(concrete_word_intersection))

	intersection_words = word_vector_words.intersection(concreteness_words)

	X_y_pairs = []

	for word in intersection_words:
		X_y_pairs.append((word_vectors[word_stoi[word]], round(ratings[concreteness_stoi[word]])))

	random.shuffle(X_y_pairs)
	split_idx = int(len(X_y_pairs) * .9)
	train_X_y_pairs = X_y_pairs[:split_idx]
	test_X_y_pairs = X_y_pairs[split_idx:]

	train_X, train_y = zip(*train_X_y_pairs)
	test_X, test_y = zip(*test_X_y_pairs)

	model = MLPClassifier(early_stopping=True)
	model.fit(train_X, train_y)

	train_y_pred = model.predict(train_X)
	test_y_pred = model.predict(test_X)

	print('Train accuracy', accuracy_score(train_y, train_y_pred))

	print('Train Confusion Matrix\n', confusion_matrix(train_y, train_y_pred))

	print('Test accuracy', accuracy_score(test_y, test_y_pred))

	print('Test Confusion Matrix\n', confusion_matrix(test_y, test_y_pred))

	prf = precision_recall_fscore_support(test_y, test_y_pred)

	print('Test Precision:', prf[0][1])
	print('Test Recall:', prf[1][1])
	print('Test F-Score:', prf[2][1])


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Runs concreteness prediction.')
	parser.add_argument('filename',
	                    help='Input embeddings txt in GloVe txt format.')

	parser.add_argument('pkl_path',
	                    help='Intermediate embeddings pkl path.')

	args = parser.parse_args()
	main(args)