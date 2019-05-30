'''
These loss functions take a matrix of similarities as output by
the functions in similarity_functions.py and output a scalar loss
value.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictionSoftmaxLoss(nn.Module):
	'''
	The loss is the cross-entropy loss of the prediction task where
	the left input is used to predict the right input if 
	predict_right=True, otherwise the prediction task where the right
	input is used to predict the left input.
	'''
	def __init__(self, predict_right=True):
		super(PredictionSoftmaxLoss, self).__init__()
		self.predict_right = predict_right

	def forward(self, similarities):
		batch_size = similarities.shape[0]
		target = torch.arange(0, batch_size)

		if not self.predict_right:
			similarities = similarities.transpose(0, 1)

		criterion = nn.CrossEntropyLoss(reduction='sum')
		
		return criterion(similarities, target)


# class TripletLoss(nn.Module):
# 	'''
# 	The triplet loss 
# 	'''
# 	def __init__(self, compare_with_left=False, compare_with_right=True, alpha=.1):
# 		if not compare_with_left and not compare_with_right:
# 			raise ValueError('One or both of compare_with_left and compare_with_right have to be True.')
# 		self.compare_with_left = compare_with_left
# 		self.compare_with_right = compare_with_right
# 		self.alpha = alpha

# 	def forward(self, similarities):
# 		batch_size = similarities.shape[0]
# 		# Randomly choose negative indices, since probability that negative
# 		# indices overlap with the correct anchors is low for large enough
# 		# batch sizes.
# 		neg_indices = torch.randint(0, batch_size, batch_size)
# 		loss1 = F.relu(alpha)
