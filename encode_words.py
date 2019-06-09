import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import karantools as kt
import encoders
import utils
import pickle

from torchtext.vocab import GloVe

VEC_SIZE = 300

glove = GloVe(name='6B', dim=VEC_SIZE)

def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

def main():
	words = sorted(glove.stoi.keys())

	left_encoder = encoders.FCEncoder((300, 300))
	right_encoder = encoders.FCEncoder((2048, 300))

	pair_encoder = encoders.PairEncoder(left_encoder, right_encoder)

	pair_encoder.load_state_dict(torch.load('outputs/left_pred/pair_encoder.pkl'))
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

	with open('outputs/left_pred/encoded_words.pkl', 'wb') as f:
		pickle.dump([aligned_stoi, aligned_vectors], f)

	utils.stoi_vectors_to_txt(aligned_stoi, aligned_vectors, 'outputs/left_pred/encoded_words.txt')

if __name__=='__main__':
	main()