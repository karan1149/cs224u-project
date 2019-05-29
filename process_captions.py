import argparse
import pickle
import karantools as kt
from torchtext.vocab import GloVe
from collections import defaultdict
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Process captions from Flickr30k.')
parser.add_argument('output_file',
                    help='Output filename for caption data.')

args = parser.parse_args()

results_path = 'flickr30k_data/results.csv'
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
	
	with open(results_path, 'r') as f:
		next(iter(f))

		for line in tqdm(f):
			line = line.strip()
			filename, index_string, caption = line.split('|')

			filename = filename.strip()
			index = int(index_string.strip())
			caption_tokens = caption.strip().split()

			average_embedding = np.zeros(VEC_SIZE)
			found_words = 0.0
			total_words = 0.0

			for token in caption_tokens:
				vec = get_word_vector(token)
				if vec is not None:
					average_embedding += vec
					found_words += 1
				total_words += 1

			if found_words:
				average_embedding /= found_words

			caption_data[filename].append(average_embedding)

	with open(args.output_file, 'wb') as f:
		pickle.dump(caption_data, f)


if __name__=='__main__':
	main()