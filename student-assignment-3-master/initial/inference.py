import numpy as np

def inference(model, input):
    """
    Do forward propagation through the network to get the activation
    at each layer, and the final output
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
    Returns:
        output: The final output of the model
        activations: A list of activations for each layer in model["layers"]
    """

    num_layers = len(model['layers'])
    activations = [None,] * num_layers

    # TODO: FORWARD PROPAGATION CODE
    activations[0], _, _ = model['layers'][0]['fwd_fn'](input, model['layers'][0]['params'], model['layers'][0]['hyper_params'], False)
    for layer_index in range(1, num_layers):
        activations[layer_index], _, _ = model['layers'][layer_index]['fwd_fn'](activations[layer_index - 1], model['layers'][layer_index]['params'], model['layers'][layer_index]['hyper_params'], False)

    output = activations[-1]
    return output, activations
