import argparse
import pickle
import karantools as kt
from torchtext.vocab import GloVe
from collections import defaultdict
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Process captions from Flickr30k.')

args = parser.parse_args()

results_path = 'flickr30k_images/results.csv'
caption_data_path = 'outputs/caption_data.pkl'
raw_captions_path = 'outputs/raw_captions.pkl'
VEC_SIZE = 300

def main():
	glove = GloVe(name='6B', dim=VEC_SIZE)

	# Returns word vector for word if it exists, else return None.
	def get_word_vector(word):
	    try:
	      return glove.vectors[glove.stoi[word.lower()]].numpy()
	    except KeyError:
	      return None

	# Maps from image file name (e.g. "1000092795.jpg") to a list of 5 numpy 
	# arrays containing average word embeddings for the words in each caption.
	caption_data = defaultdict(list)
	raw_captions = defaultdict(list)
	
	with open(results_path, 'r') as f:
		next(iter(f))

		for line in tqdm(f):
			line = line.strip()
			filename, index_string, caption = line.split('|')

			filename = filename.strip()
			index = int(index_string.strip())
			caption_tokens = caption.strip().split()
			found_tokens = []

			average_embedding = np.zeros(VEC_SIZE)
			found_words = 0.0
			total_words = 0.0

			for token in caption_tokens:
				vec = get_word_vector(token)
				if vec is not None:
					average_embedding += vec
					found_words += 1
					found_tokens.append(token)
				total_words += 1

			if found_words:
				average_embedding /= found_words

			raw_captions[filename].append(found_tokens)

			caption_data[filename].append(average_embedding)

	with open(caption_data_path, 'wb') as f:
		pickle.dump(caption_data, f)

	with open(raw_captions_path, 'wb') as f:
		pickle.dump(raw_captions, f)

if __name__=='__main__':
	main()