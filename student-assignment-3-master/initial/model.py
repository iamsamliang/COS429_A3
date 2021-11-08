"""
    Usage:
        python model.py [use_saved_model]
        
        use_saved_model can only be "true" or "false (capital-sensitive)
"""


import sys
sys.path += ['layers']
import numpy as np
from init_layers import init_layers
from init_model import init_model
from train import train
from data_utils import get_CIFAR10_data
import matplotlib.pyplot as plt
import sys
import pickle

def main():
    
    # saved model
    if sys.argv[1] == "true":
        use_saved_model = True
    elif sys.argv[1] == "false":
        use_saved_model = False
    else:
        sys.exit("Usage: python model.py [use_saved_model]")

    X_train, y_train, X_test, y_test = get_CIFAR10_data()

    if use_saved_model:
        model = np.load('model.npz', allow_pickle=True)
        model = dict(model)
        layers = model["layers"]
        
        with open('metrics.pickle', 'rb') as file:
            metrics = pickle.load(file)
        
    else:
        # after conv1, relu, and batchnorm: 30x30x8
        conv1 = init_layers('conv', {'filter_size': 3, 'filter_depth': 3, 'num_filters': 8})
        batchnorm1 = init_layers('batchnorm', {'in_height': 30, 'in_width': 30, 'num_channels': 8})
        # after conv2, relu, and batchnorm: 28x28x10
        conv2 = init_layers('conv', {'filter_size': 3, 'filter_depth': 8, 'num_filters': 10})
        batchnorm2 = init_layers('batchnorm', {'in_height': 28, 'in_width': 28, 'num_channels': 10})
        relu = init_layers('relu', {})
        # after pool: 14x14x10
        pool = init_layers('pool', {'filter_size': 2, 'stride': 2})
        # after flatten: 1960
        flatten = init_layers('flatten', {})
        linear1 = init_layers('linear', {'num_in': 1960, 'num_out': 140})
        linear2 = init_layers('linear', {'num_in': 140, 'num_out': 10})
        softmax = init_layers('softmax', {})

        layers = [conv1, relu, batchnorm1,
                  conv2, relu, batchnorm2,
                  pool, flatten,
                  linear1, relu,
                  linear2, softmax]

        metrics = None

        model = init_model(layers, [X_train.shape[0], X_train.shape[1], X_train.shape[2]], 10, True)
    
    numIters = 3
    params = {"learning_rate": 1e-2, 
              "weight_decay": 1e-3,
              "batch_size": 128,
              "friction_rho": 0.99,
              "eps": 1e-5,
              "metrics": metrics}
    
    model = train(model, X_train, y_train, X_test, y_test, params, numIters)
    
    # reporting the results
    
    # loss-over-epoch
    plt.plot(params["metrics"]["training_metrics"]["training_recording_epochs"], params["metrics"]["training_metrics"]["training_losses"], color="red", label="Training Loss")
    plt.plot(params["metrics"]["testing_metrics"]["testing_recording_epochs"], params["metrics"]["testing_metrics"]["testing_losses"], color="blue", label="Testing Loss")
    plt.title("Cross-Entropy Loss over Training Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss/loss_{curr_epoch:.4f}.png".format(curr_epoch=params["metrics"]["curr_epoch"]))
    plt.close()
    
    # accuracy over epoch
    plt.plot(params["metrics"]["training_metrics"]["training_recording_epochs"], params["metrics"]["training_metrics"]["training_accs"], color="red", label="Training Accuracy")
    plt.plot(params["metrics"]["testing_metrics"]["testing_recording_epochs"], params["metrics"]["testing_metrics"]["testing_accs"], color="blue", label="Testing Accuracy")
    plt.title("Classification Accuracy over Training Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("acc/acc_{curr_epoch:.4f}.png".format(curr_epoch=params["metrics"]["curr_epoch"]))
    plt.close()
    
if __name__ == '__main__':
    main()