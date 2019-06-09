import pickle
import karantools as kt
import math
from torchtext.vocab import GloVe
from tqdm import tqdm

import numpy as np
import random

VEC_SIZE = 300

NUM_CAPTIONS_PER_IMAGE = 5

IMAGE_FEATURES_SIZE = 2048

ENRICHED_WORD_VECTORS_PATH = 'outputs/encoded_words.pkl'

def max_tf_idf_term(raw_captions, filename, idx, idfs):
	terms = raw_captions[filename][idx]
	unique_words = sorted(list(set(terms)))

	tfidf_vec = np.zeros(len(unique_words))

	for i, term in enumerate(unique_words):
		tf = terms.count(term)
		
		idf = get_idf(raw_captions, term, idfs)
		tfidf_vec[i] = tf / float(len(terms)) * idf

	max_i = np.argmax(tfidf_vec)
	return unique_words[max_i]

def get_idf(raw_captions, term, idfs):
	if term in idfs:
		return idfs[term]

	num_docs_with_term = 0.0

	for image in raw_captions:
		for idx in range(NUM_CAPTIONS_PER_IMAGE):
			if term in raw_captions[image][idx]:
				num_docs_with_term += 1

	idfs[term] = math.log(len(raw_captions) * NUM_CAPTIONS_PER_IMAGE / num_docs_with_term)
	return idfs[term]


def main():
	RAW_CAPTIONS_PATH = 'outputs/raw_captions.pkl'
	CNN_EMBEDDINGS_PATH = 'outputs/cnn_embeddings.pkl'
	
	with open(RAW_CAPTIONS_PATH, 'rb') as f:
	    raw_captions = pickle.load(f)

	with open(CNN_EMBEDDINGS_PATH, 'rb') as f:
	    cnn_embeddings = pickle.load(f)

	with open(ENRICHED_WORD_VECTORS_PATH, 'rb') as f:
		enriched_stoi, enriched_vectors = pickle.load(f)

	images = sorted(raw_captions.keys())
	random.shuffle(images)
	images = images[:1000]
	print('Num images', len(images))

	glove = GloVe(name='6B', dim=VEC_SIZE)

	# Returns word vector for word if it exists, else return None.
	def get_word_vector(word):
	    try:
	      return glove.vectors[glove.stoi[word.lower()]].numpy()
	    except KeyError:
	      return None

	def get_enriched_word_vector(word):
		try:
		  return enriched_vectors[enriched_stoi[word.lower()]]
		except KeyError:
		  return None

	try:
		with open('outputs/caption_idfs.pkl', 'rb') as f:  
			idfs = pickle.load(f)
	except:
		idfs = {}

	glove_X = np.zeros((2 * NUM_CAPTIONS_PER_IMAGE * len(images), VEC_SIZE + IMAGE_FEATURES_SIZE))
	glove_y = np.zeros(2 * NUM_CAPTIONS_PER_IMAGE * len(images))


	enriched_X = np.zeros((2 * NUM_CAPTIONS_PER_IMAGE * len(images), VEC_SIZE + IMAGE_FEATURES_SIZE))
	enriched_y = np.zeros(2 * NUM_CAPTIONS_PER_IMAGE * len(images))

	vocab = list(glove.stoi.keys())

	for image_i, image in tqdm(enumerate(images), total=len(images)):
		for idx in range(NUM_CAPTIONS_PER_IMAGE):
			max_term = max_tf_idf_term(raw_captions, image, idx, idfs)
			random_term = random.choice(vocab)

			glove_X[image_i * 10 + idx * 2] = np.concatenate([get_word_vector(max_term), cnn_embeddings[image]])
			glove_X[image_i * 10 + idx * 2 + 1] = np.concatenate([get_word_vector(random_term), cnn_embeddings[image]])

			glove_y[image_i * 10 + idx * 2] = 1
			glove_y[image_i * 10 + idx * 2 + 1] = 0

			enriched_X[image_i * 10 + idx * 2] = np.concatenate([get_enriched_word_vector(max_term), cnn_embeddings[image]])
			enriched_X[image_i * 10 + idx * 2 + 1] = np.concatenate([get_enriched_word_vector(random_term), cnn_embeddings[image]])

			enriched_y[image_i * 10 + idx * 2] = 1
			enriched_y[image_i * 10 + idx * 2 + 1] = 0


	with open('outputs/relatedness_glove_data.pkl', 'wb') as f:
		pickle.dump([glove_X, glove_y], f)

	with open('outputs/relatedness_enriched_data.pkl', 'wb') as f:
		pickle.dump([enriched_X, enriched_y], f)

	with open('outputs/caption_idfs.pkl', 'wb') as f:
		pickle.dump(idfs, f)

if __name__=='__main__':
	main()