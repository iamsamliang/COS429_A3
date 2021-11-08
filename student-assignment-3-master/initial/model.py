import sys
sys.path += ['layers']
import numpy as np
from init_layers import init_layers
from init_model import init_model
from train import train
from data_utils import get_CIFAR10_data

def main():
    use_saved_model = False
    X_train, y_train, X_test, y_test = get_CIFAR10_data()
    
    # after conv1 and relu: 30x30x8
    conv1 = init_layers('conv', {'filter_size': 3, 'filter_depth': 3, 'num_filters': 8})
    # after conv2 and relu: 28x28x10
    conv2 = init_layers('conv', {'filter_size': 3, 'filter_depth': 8, 'num_filters': 10})
    relu = init_layers('relu', {})
    # after pool: 14x14x10
    pool = init_layers('pool', {'filter_size': 2, 'stride': 2})
    # after flatten: 1960
    flatten = init_layers('flatten', {})
    linear = init_layers('linear', {'num_in': 1960, 'num_out': 10})
    softmax = init_layers('softmax', {})
    
    layers = [conv1, relu, conv2, relu, pool, flatten, linear, softmax]

    model = init_model(layers, [X_train.shape[0], X_train.shape[1], X_train.shape[2]], 10, True)
    if use_saved_model:
        model = np.load('model.npz', allow_pickle=True)
        model = dict(model)
    
    numIters = 400
    params = {"learning_rate": 1e-2, 
              "weight_decay": 1e-2,
              "batch_size": 128,
              "friction_rho": 0.99,
              "eps": 1e-5}
    model, loss = train(model, X_train, y_train, X_test, y_test, params, numIters)
    print(loss)

if __name__ == '__main__':
    main()