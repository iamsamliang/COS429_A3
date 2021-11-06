import sys
sys.path += ['layers']
import numpy as np
from init_layers import init_layers
from init_model import init_model
from train import train
from data_utils import get_CIFAR10_data

def main():
    X_train, y_train, X_test, y_test = get_CIFAR10_data()
    
    layers = [init_layers('conv', {'filter_size': 2,
                              'filter_depth': 3,
                              'num_filters': 2}),
         init_layers('pool', {'filter_size': 2,
                              'stride': 2}),
         init_layers('relu', {}),
         init_layers('flatten', {}),
         init_layers('linear', {'num_in': 32,
                                'num_out': y_train.shape[0]}),
         init_layers('softmax', {})]

    model = init_model(layers, [X_train.shape[0], X_train.shape[1], X_train.shape[2]], y_train.shape[0], True)
    
    numIters = 10
    params = {}
    model, loss = train(model, X_train, y_train, params, numIters)
    print(loss)

if __name__ == '__main__':
    main()