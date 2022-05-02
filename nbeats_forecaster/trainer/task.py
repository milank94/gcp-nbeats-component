import argparse
import os

import tensorflow as tf

from trainer import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Vertex custom container training args. These are set by Vertex AI during training but can also be overwritten.
    parser.add_argument('--model-dir', dest='model-dir',
                        default=os.environ['AIP_MODEL_DIR'], type=str, help='GCS URI for saving model artifacts.')
    parser.add_argument('--train_dataset', dest='train_dataset', default=os.environ['TRAIN_DIR'],
                        type=str, help='Location of training dataset.')
    parser.add_argument('--test_dataset', dest='test_dataset', default=os.environ['TEST_DIR'],
                        type=str, help='Location of test dataset.')

    # Model training args.
    parser.add_argument('--input_size', dest='input_size', default=7, type=int, help='Input data size.')
    parser.add_argument('--theta_size', dest='theta_size', default=8, type=int,
                        help='Theta parameter for N-BEATS model.')
    parser.add_argument('--n_epochs', dest='n_epochs', default=5000, type=int, help='Training iterations.')
    parser.add_argument('--horizon', dest='horizon', default=1, type=int, help='Forecasting horizon.')
    parser.add_argument('--n_neurons', dest='n_neurons', default=512, type=int,
                        help='Number of neurons per hidden layer.')
    parser.add_argument('--n_layers', dest='n_layers', default=4, type=int, help='Number of hidden layers.')
    parser.add_argument('--n_stacks', dest='n_stacks', default=30, type=int, help='Number of stacks.')

    args = parser.parse_args()
    hparams = args.__dict__

    train_dataset = tf.data.TFRecordDataset([hparams['train_dataset']])
    test_dataset = tf.data.TFRecordDataset([hparams['test_dataset']])

    model.train_evaluate(hparams=hparams, train_dataset=train_dataset, test_dataset=test_dataset)
