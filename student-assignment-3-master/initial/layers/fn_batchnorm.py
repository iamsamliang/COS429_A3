import numpy as np

def fn_batchnorm(input, params, hyper_params, backprop, dv_output=None):
    """
    Args:
        input: The input data to the layer function. [in_height] x [in_width] x [num_channels] x [batch_size] array
        params: Gamma and beta information for the affine transform in this layer's batch normalization (y = gamma*x + beta). 
            params['W']: layer slope (gamma), same as the first three dimensions of input
            params['b']: layer bias (beta), same as the first three dimensions of input
        hyper_params: Dummy variable, not used.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, same size as input
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: The gradient term that you will use to update the weights defined in params and train your network. Dictionary with same structure as params.
            grad['W']: gradient wrt to the slope, same size as params['gamma']
            grad['b']: gradient wrt bias, same size as params['beta']
    """

    in_height, in_width, num_channels, batch_size = input.shape
    epsilon = 1e-4 # used for numerical stability
    
    # Initialize
    output = np.zeros(input.shape)
    dv_input = np.zeros(0)
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}
    
    normalized_input = np.zeros(input.shape)
    y = np.zeros(input.shape)
    
    # FORWARD CODE

    means = np.mean(input, axis=3, keepdims=True)
    variances = np.var(input, axis=3, keepdims=True)

    normalized_input = (input - means) / np.sqrt(variances + epsilon)
    output = params["W"]*normalized_input + params["b"]
    
    if backprop:
        assert dv_output is not None
        dv_input = np.zeros(input.shape)
        grad['W'] = np.zeros(params['W'].shape)
        grad['b'] = np.zeros(params['b'].shape)
        
        # BACKPROP CODE
        #       Update dv_input and grad with values
        # analytical derivations can be found on Wikipedia
        
        dv_normalized_input = dv_output * params["W"]
        dv_variances = - np.sum(dv_output*(input - means)*params["W"]*np.power(variances + epsilon, -1.5)/2 , axis=3, keepdims=True)
        dv_means = - np.sum(dv_output*params["W"]/np.sqrt(variances + epsilon), axis=3, keepdims=True) - 2*dv_variances*np.sum(input - means, axis=3, keepdims=True)/batch_size       
        dv_input = dv_normalized_input / np.sqrt(variances + epsilon) + dv_means / batch_size + 2*dv_variances*(input - means)/batch_size
        
    return output, dv_input, grad
