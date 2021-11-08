import numpy as np
import scipy.signal

def fn_conv(input, params, hyper_params, backprop, dv_output=None):
    """
    Args:
        input: The input data to the layer function. [in_height] x [in_width] x [num_channels] x [batch_size] array
        params: Weight and bias information for the layer.
            params['W']: layer weights, [filter_height] x [filter_width] x [filter_depth] x [num_filters] array
            params['b']: layer bias, [num_filters] x 1 array
        hyper_params: Optional, could include information such as stride and padding.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [out_height] x [out_width] x [num_filters] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: The gradient term that you will use to update the weights defined in params and train your network. Dictionary with same structure as params.
            grad['W']: gradient wrt weights, same size as params['W']
            grad['b']: gradient wrt bias, same size as params['b']
    """
    
    print("Starting Conv Layer")

    in_height, in_width, num_channels, batch_size = input.shape
    _, _, filter_depth, num_filters = params['W'].shape
    out_height = in_height - params['W'].shape[0] + 1
    out_width = in_width - params['W'].shape[1] + 1

    assert params['W'].shape[2] == input.shape[2], 'Filter depth does not match number of input channels'

    # Initialize
    output = np.zeros((out_height, out_width, num_filters, batch_size))
    dv_input = np.zeros(0)
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}
    
    # TODO: FORWARD CODE
    #       Update output with values
    
    # depth-flipped filters are used for implementing the 2d convolution + depth-sum as 3d convolution.
    depth_flipped_filters = np.flip(params['W'], axis=2)
    
    # apply 3d convolution to every im in the batch to with every filter.
    for im_num in range(batch_size):
        for filter_k in range(num_filters):
            output[:, :, filter_k, im_num] = scipy.signal.convolve(input[:, :, :, im_num], depth_flipped_filters[:, :, :, filter_k], mode="valid").reshape(out_height, out_width) + params['b'][filter_k]

    if backprop:
        assert dv_output is not None
        dv_input = np.zeros(input.shape)
        grad['W'] = np.zeros(params['W'].shape)
        grad['b'] = np.zeros(params['b'].shape)
        
        # TODO: BACKPROP CODE
        #       Update dv_input and grad with values
        
        # flipped inputs and filters are needed
        backprop_flipped_filters = np.flip(depth_flipped_filters, axis=(0,1,2)) # don't flip the [num_filters] dimension
        backprop_flipped_inputs = np.flip(input, axis=(0,1,2)) # don't flip the [batch_size] dimension
                                          
        
        # grad['b']: for a given depth-level k of the 4-dimensional output, every entry (per width per height)
        # contributes 1 to the derivative due to the element-wise nature of summation. 
        # then, the average over the batch is found.
        for filter_k in range(num_filters):
            grad['b'][filter_k] += np.sum(dv_output[:, :, filter_k, :])
        grad['b'] = grad['b'] / batch_size
        
        # grad['W']: the gradient of the loss w.r.t. the weights corresponds to taking a convolution between
        # the flipped image and the gradients of the loss w.r.t. the outputs
        for filter_k in range(num_filters):
            for im_num in range(batch_size):
                grad['W'][:, :, :, filter_k] += scipy.signal.convolve(backprop_flipped_inputs[:, :, :, im_num], dv_output[:, :, filter_k, im_num].reshape(out_height, out_width, 1), mode='valid')
                
        # need to take the average over every batch, also need to flip again before returning
        grad['W'] = grad['W'] / batch_size
        grad['W'] = np.flip(grad['W'], axis=2)
        
        # dv_input: the gradient of the loss w.r.t. the input corresponds to taking a convolution between
        # the flipped filters and the gradients of the loss w.r.t. the outputs
        for filter_k in range(num_filters):
            for im_num in range(batch_size):
                for channel in range(num_channels):
                    dv_input[:,:,channel,im_num] += scipy.signal.convolve(backprop_flipped_filters[:, :, channel, filter_k], dv_output[:, :, filter_k, im_num], mode='full')

    print("Finished with Conv Layer")
    return output, dv_input, grad
