import numpy as np

def calc_gradient(model, input, layer_acts, dv_output):
    '''
    Calculate the gradient at each layer, to do this you need dv_output
    determined by your loss function and the activations of each layer.
    The loop of this function will look very similar to the code from
    inference, just looping in reverse.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
        layer_acts: A list of activations of each layer in model["layers"]
        dv_output: The partial derivative of the loss with respect to each element in the output matrix of the last layer.
    Returns: 
        grads:  A list of gradients of each layer in model["layers"]
    '''
    num_layers = len(model["layers"])
    grads = [None,] * num_layers

    # TODO: Determine the gradient at each layer.
    #       Remember that back-propagation traverses 
    #       the model in the reverse order.
    for layer_index in reversed(range(1, num_layers)):
        print(layer_index)
        curr_layer = model['layers'][layer_index]
        print(curr_layer['fwd_fn'])
        print(curr_layer['params'])
        print(curr_layer['hyper_params'])
        curr_activations = layer_acts[layer_index - 1]
        _, dv_output, grad = curr_layer['fwd_fn'](curr_activations, curr_layer['params'], curr_layer['hyper_params'], True, 
                                                  dv_output=dv_output)
        grads[layer_index] = grad
    
    _, dv_output, grad = model['layers'][0]['fwd_fn'](input, model['layers'][0]['params'], model['layers'][0]['hyper_params'], 
                                                      True, dv_output=dv_output)
    grads[0] = grad
        

    return grads