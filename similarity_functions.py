'''
Functions for computing similarities over two batches of outputs, 
left_encodings and right_encodings. All of these functions return 
a matrix S of pairwise similarities, where S[i][j] is the similarity
between the ith left encoding and the jth right encoding.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_similarity(left_encodings, right_encodings):
	left_encodings_normalized = left_encodings / left_encodings.norm(dim=1)[:, None]
	right_encodings_normalized = right_encodings / right_encodings.norm(dim=1)[:, None]

	return left_encodings_normalized @ right_encodings_normalized.transpose(0, 1)

def dot_product(left_encodings, right_encodings):
	return left_encodings_normalized @ right_encodings_normalized.transpose(0, 1)