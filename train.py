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
import functools
import os

# parser = argparse.ArgumentParser(description='Train alignment model for GloVe word embeddings on Flickr30k data.')
# # parser.add_argument('',
# #                     help='')
# parser.add_argument('--cuda', action='store_true',
#                     help='Whether to use GPU or not.')

# args = parser.parse_args()

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

print('Training on %s...' % DEVICE)

NUM_EPOCHS = 40
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4

TRAIN_FRACTION = 0.95

def main(basepath):
    train_data, val_data = kt.lazy_load(functools.partial(utils.load_caption_image_data, train_fraction=TRAIN_FRACTION), os.path.join(basepath, 'outputs/caption_image_data.pkl'))   

    train_caption_matrix, train_image_matrix, train_idx_dict = train_data   
    val_caption_matrix, val_image_matrix, val_idx_dict = val_data

    left_encoder = encoders.FCEncoder((300, 300))
    right_encoder = encoders.FCEncoder((2048, 300))

    kt.assert_eq(len(utils.compute_output_shape(left_encoder, (1, 300))), 2)
    kt.assert_eq(len(utils.compute_output_shape(right_encoder, (1, 2048))), 2)
    kt.assert_eq(utils.compute_output_shape(left_encoder, (1, 300)), utils.compute_output_shape(right_encoder, (1, 2048)))

    pair_encoder = encoders.PairEncoder(left_encoder, right_encoder)

    print(pair_encoder)

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

            with torch.set_grad_enabled(phase == 'train'):

                running_loss = 0.0

                curr_data = train_data if phase == 'train' else val_data

                num_batches = 0
                for left_x, right_x in utils.flickr_dataloader(curr_data, BATCH_SIZE):
                    num_batches += 1

                    left_x.to(DEVICE)
                    right_x.to(DEVICE)

                    left_encodings, right_encodings = pair_encoder(left_x, right_x)

                    similarities = similarity_functions.dot_product(left_encodings, right_encodings)


                    criterion = loss_functions.PredictionSoftmaxLoss(predict_right=True)

                    loss = criterion(similarities)
                    running_loss += loss.item()

                    if phase == 'train':
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
    plt.title('Loss: Dot Product + Softmax Right Prediction Task')
    plt.legend(fancybox=True, framealpha=0.5)
    plt.savefig('outputs/losses.png')

    torch.save(pair_encoder.state_dict(), 'outputs/pair_encoder.pkl')

if __name__=='__main__':
    main('.')