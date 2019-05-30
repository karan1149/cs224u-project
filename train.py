import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
import pickle
import argparse
import karantools as kt
import utils
import encoders
import similarity_functions
import loss_functions
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Train alignment model for GloVe word embeddings on Flickr30k data.')
# parser.add_argument('',
#                     help='')
parser.add_argument('--cuda', action='store_true',
                    help='Whether to use GPU or not.')

args = parser.parse_args()

CAPTION_DATA_PATH = 'outputs/caption_data.pkl'
CNN_EMBEDDINGS_PATH = 'outputs/cnn_embeddings.pkl'
DEVICE = 'cuda' if args.cuda else 'cpu'

NUM_EPOCHS = 30
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4

TRAIN_FRACTION = 0.90

def main():
    with open(CAPTION_DATA_PATH, 'rb') as f:
        caption_data = pickle.load(f)

    with open(CNN_EMBEDDINGS_PATH, 'rb') as f:
        cnn_embeddings = pickle.load(f)

    # def transform_caption_cnn_data():
    #     cnn_embeddings

    caption_data = {k:torch.as_tensor(caption_data[k]) for k in caption_data}
    cnn_embeddings = {k:torch.as_tensor(cnn_embeddings[k]) for k in cnn_embeddings}

    images = sorted(caption_data.keys())

    kt.assert_eq(images, sorted(cnn_embeddings.keys()))

    left_encoder = encoders.FCEncoder((300, 300))
    right_encoder = encoders.FCEncoder((2048, 300))

    kt.assert_eq(len(utils.compute_output_shape(left_encoder, (1, 300))), 2)
    kt.assert_eq(len(utils.compute_output_shape(right_encoder, (1, 2048))), 2)
    kt.assert_eq(utils.compute_output_shape(left_encoder, (1, 300)), utils.compute_output_shape(right_encoder, (1, 2048)))

    pair_encoder = encoders.PairEncoder(left_encoder, right_encoder)

    print(pair_encoder)

    split_idx = int(len(images) * TRAIN_FRACTION)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    train_example_indices = utils.get_example_indices(train_images)
    val_example_indices = utils.get_example_indices(val_images)

    optimizer = torch.optim.Adam(pair_encoder.parameters(), lr=LEARNING_RATE)

    running_losses = defaultdict(list)

    for epoch_i in range(NUM_EPOCHS):
        print('\n' + '-' * 20)
        print('Epoch %d of %d' % (epoch_i, NUM_EPOCHS))
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                pair_encoder.train()  # Set model to training mode
            else:
                pair_encoder.eval()   # Set model to evaluate mode
        
            running_loss = 0.0

            curr_example_indices = train_example_indices if phase == 'train' else val_example_indices

            num_batches = 0
            for left_x, right_x in utils.flickr_dataloader(curr_example_indices, caption_data, cnn_embeddings, BATCH_SIZE):
                num_batches += 1

                left_x.to(DEVICE)
                right_x.to(DEVICE)

                left_encodings, right_encodings = pair_encoder(left_x, right_x)

                similarities = similarity_functions.dot_product(left_encodings, right_encodings)


                criterion = loss_functions.PredictionSoftmaxLoss(predict_right=True)

                loss = criterion(similarities)
                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss /= num_batches

            print(phase, 'loss', running_loss)
            running_losses[phase].append(running_loss)

    plt.plot(running_losses['train'], label='Train')
    plt.plot(running_losses['val'], label='Val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training using Dot Product and Softmax Right Pred. Loss')
    plt.savefig('outputs/losses.png')

    torch.save(pair_encoder, 'outputs/pair_encoder.pkl')

if __name__=='__main__':
    main()