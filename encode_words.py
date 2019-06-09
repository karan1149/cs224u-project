import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import karantools as kt
import encoders
import utils
import pickle
import argparse
import json
import os

from torchtext.vocab import GloVe

VEC_SIZE = 300

glove = GloVe(name='6B', dim=VEC_SIZE)

def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

def main(config):
	words = sorted(glove.stoi.keys())

	left_encoder = encoders.FCEncoder(config['left_layer_sizes'])
	right_encoder = encoders.FCEncoder(config['right_layer_sizes'])

	pair_encoder = encoders.PairEncoder(left_encoder, right_encoder)

	MODEL_DIR = os.path.join('outputs', 'models', config['name'])

	pair_encoder.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'pair_encoder.pkl')))
	word_encoder = pair_encoder.left_encoder
	word_encoder.eval()

	aligned_stoi = {word:i for i, word in enumerate(words)}
	aligned_vectors = torch.zeros(len(words), VEC_SIZE)

	with torch.no_grad():
	
		for i, word in tqdm(enumerate(words), total=len(words)):
			word_vec = get_word_vector(word)

			word_tensor = torch.unsqueeze(torch.as_tensor(word_vec), 0)

			kt.assert_eq(word_tensor.shape, (1, VEC_SIZE))

			aligned_vectors[i] = word_encoder(word_tensor)

	aligned_vectors = aligned_vectors.numpy()

	with open(os.path.join(MODEL_DIR, 'encoded_words.pkl'), 'wb') as f:
		pickle.dump([aligned_stoi, aligned_vectors], f)

	utils.stoi_vectors_to_txt(aligned_stoi, aligned_vectors, os.path.join(MODEL_DIR, 'encoded_words.txt'))

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Encodes word embeddings using pretrained pair encoder.')
	parser.add_argument('config_path',
	                    help='')

	args = parser.parse_args()
	with open(args.config_path, 'r') as f:
	    config = json.load(f)
	main(config)