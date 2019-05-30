import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
	print(words[:10])
	print(len(words))
	


if __name__=='__main__':
	main()