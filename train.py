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
import json

def main(config):

    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda'

    print('Training on %s...' % DEVICE)


    NUM_EPOCHS = config['num_epochs']
    BATCH_SIZE = config['batch_size']
    LEARNING_RATE = config['learning_rate']

    TRAIN_FRACTION = config['train_fraction']

    MODEL_DIR = os.path.join('outputs', 'models', config['name'])

    train_data, val_data = kt.lazy_load(functools.partial(utils.load_caption_image_data, train_fraction=TRAIN_FRACTION), 'outputs/caption_image_data.pkl')

    train_caption_matrix, train_image_matrix, train_idx_dict = train_data   
    val_caption_matrix, val_image_matrix, val_idx_dict = val_data

    left_encoder = encoders.FCEncoder(config['left_layer_sizes'])
    right_encoder = encoders.FCEncoder(config['right_layer_sizes'])

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

                    similarities = getattr(similarity_functions, config['similarity_function'])(left_encodings, right_encodings)

                    loss_args = {}
                    if config['loss_function'] == 'PredictionSoftmaxLoss':
                        loss_args['predict_right'] = config['predict_right']

                    criterion = getattr(loss_functions, config['loss_function'])(**loss_args)

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
    plt.savefig(os.path.join(MODEL_DIR, 'losses.png'))

    torch.save(pair_encoder.state_dict(), os.path.join(MODEL_DIR, 'pair_encoder.pkl'))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train alignment model for GloVe word embeddings on Flickr30k data.')
    parser.add_argument('config_path',
                        help='')

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    main(config)