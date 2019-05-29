import torch
import torch.nn as nn
import torch.nn.functional as F

class FCEncoder(nn.Module):
	def __init__(self, layer_sizes):
		super(FCEncoder, self).__init__()
		self.layers = []
		for i in range(len(layer_sizes) - 1):
			self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
		self.layers = nn.ModuleList(self.layers)

	def forward(self, x):
		for layer in self.layers:
			x = F.relu(layer(x))

		return x

class PairEncoder(object):
	def __init__(self, left_encoder, right_encoder):
		self.left_encoder = left_encoder
		self.right_encoder = right_encoder

	def forward(self, left_x, right_x):
		left_x = self.left_encoder(left_x)
		right_x = self.right_encoder(right_x)
		return left_x, right_x
