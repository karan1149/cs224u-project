import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import pickle
import karantools as kt
from tqdm import tqdm

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

def load_caption_image_data(train_fraction=0.9):
    CAPTION_DATA_PATH = 'outputs/caption_data.pkl'
    CNN_EMBEDDINGS_PATH = 'outputs/cnn_embeddings.pkl'
    
    with open(CAPTION_DATA_PATH, 'rb') as f:
        caption_data = pickle.load(f)

    with open(CNN_EMBEDDINGS_PATH, 'rb') as f:
        cnn_embeddings = pickle.load(f)

    caption_size = caption_data[next(iter(caption_data))][0].shape[0]
    cnn_embeddings_size = cnn_embeddings[next(iter(cnn_embeddings))].shape[0]

    images = sorted(caption_data.keys())
    num_images = len(images)

    split_idx = int(len(images) * train_fraction)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    train_example_indices = get_example_indices(train_images)
    val_example_indices = get_example_indices(val_images)

    split_datas = []

    for split_example_indices in [train_example_indices, val_example_indices]:

        caption_matrix = torch.zeros(len(split_example_indices), caption_size)
        image_matrix = torch.zeros(len(split_example_indices), cnn_embeddings_size)

        idx_dict = {}

        for i, (image, caption_idx) in enumerate(split_example_indices):
            caption_matrix[i] = torch.as_tensor(caption_data[image][caption_idx])
            image_matrix[i] = torch.as_tensor(cnn_embeddings[image])
            idx_dict[(image, caption_idx)] = i

        split_datas.append((caption_matrix, image_matrix, idx_dict))

    return split_datas

'''
Generator that yields batches of size batch_size with caption data on the left
and CNN embeddings/features on the right. Batches are randomly shuffled on
each run. Input data is first copied then shuffled, so it is not modified.
'''
def flickr_dataloader(curr_data, batch_size):
    caption_matrix, image_matrix, idx_dict = curr_data

    num_examples = caption_matrix.shape[0]
    permutation = np.random.permutation(num_examples)

    caption_matrix = caption_matrix[permutation]
    image_matrix = image_matrix[permutation]

    num_batches = int(np.ceil(float(num_examples) / batch_size))

    caption_size = caption_matrix.shape[1]
    image_size = image_matrix.shape[1]

    for batch_i in range(num_batches):
        curr_batch_size = batch_size

        if batch_i == num_batches - 1:
            curr_batch_size = num_examples % batch_size or batch_size

        left_x = caption_matrix[batch_i * batch_size: batch_i * batch_size + curr_batch_size]
        right_x = image_matrix[batch_i * batch_size: batch_i * batch_size + curr_batch_size]

        yield left_x, right_x
'''
Saves a set of word vectors represented by stoi, a dictionary-like object
mapping from words to indices, and vectors, a numpy array of dimensionality
(num_words, embed_dim) whose indices correspond to the ones returned by stoi.
Text is formatted in GloVe format as in:
https://radimrehurek.com/gensim/scripts/glove2word2vec.html
'''
def stoi_vectors_to_txt(stoi, vectors, txt_filename):
    words = sorted(stoi.keys(), key=lambda word: stoi[word])

    with open(txt_filename, 'w') as f:
        for i, word in tqdm(enumerate(words), total=len(words)):
            word_embedding = vectors[i].tolist()
            f.write(word + ' ' + ' '.join([str(entry) for entry in word_embedding]) + '\n')
