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
    activations[0], _, _ = model['layers']['fwd_fn'](input, layer['params'], layer['hyper_params'], False)
    for layer in range(1, num_layers):
        activations[layer_index], _, _ = model['layers']['fwd_fn'](activations[layer_index - 1], layer['params'], layer['hyper_params'], False)

    output = activations[-1]
    return output, activations
