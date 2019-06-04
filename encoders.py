import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import karantools as kt

class FCEncoder(nn.Module):
	def __init__(self, layer_sizes):
		super(FCEncoder, self).__init__()
		self.layers = []
		for i in range(len(layer_sizes) - 1):
			self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
		self.layers = nn.ModuleList(self.layers)

	def forward(self, x):
		for i, layer in enumerate(self.layers):
			if i != len(self.layers) - 1:
				x = F.relu(layer(x))
			else:
				x = layer(x)
		return x

class PairEncoder(nn.Module):
	def __init__(self, left_encoder, right_encoder):
		super(PairEncoder, self).__init__()
		self.left_encoder = left_encoder
		self.right_encoder = right_encoder

	def forward(self, left_x, right_x):
		left_x = self.left_encoder(left_x)
		right_x = self.right_encoder(right_x)
		return left_x, right_x