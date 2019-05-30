import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

def compute_output_shape(mod, input_shape):
    """
    Compute output size of Module `mod` given an input with size `in_size`.
    """
    
    f = mod.forward(autograd.Variable(torch.Tensor(1, *input_shape)))
    return f.shape[1:]

def compute_output_size(mod, input_shape):
	return int(np.prod(compute_output_shape(mod, input_shape)))

'''
Given a list of image filenames, e.g. 103295345.jpg, return a list of pairs
containing (image filename, caption idx), where caption index ranges from 0 to
4.
'''
def get_example_indices(images):
	example_indices = []
	for image in images:
		for i in range(5):
			example_indices.append((image, i))
	return example_indices

'''
Generator that yields batches of size batch_size with caption data on the left
and CNN embeddings/features on the right. Batches are randomly shuffled on each run.
'''
def flickr_dataloader(example_indices, caption_data, cnn_embeddings, batch_size):
	num_examples = len(example_indices)
	permutation = np.random.permutation(num_examples)

	num_batches = int(np.ceil(float(num_examples) / batch_size))

	caption_size = caption_data[next(iter(caption_data))][0].shape[0]
	cnn_embeddings_size = cnn_embeddings[next(iter(cnn_embeddings))].shape[0]

	for batch_i in range(num_batches):
		curr_batch_size = batch_size

		if batch_i == num_batches - 1:
			curr_batch_size = num_examples % batch_size or batch_size

		left_x = torch.zeros(curr_batch_size, caption_size)
		right_x = torch.zeros(curr_batch_size, cnn_embeddings_size)

		for j in range(curr_batch_size):
			image, caption_idx = example_indices[batch_i * batch_size + j]
			left_x[j] = caption_data[image][caption_idx]
			right_x[j] = cnn_embeddings[image]

		yield left_x, right_x
