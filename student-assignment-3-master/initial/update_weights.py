import numpy as np

def update_weights(model, grads, hyper_params):
    '''
    Update the weights of each layer in your model based on the calculated gradients
    Args:
        model: Dictionary holding the model
        grads: A list of gradients of each layer in model["layers"]
        hyper_params: 
            hyper_params['learning_rate']
            hyper_params['weight_decay']: Should be applied to W only.
    Returns: 
        updated_model:  Dictionary holding the updated model
    '''
    num_layers = len(grads)
    a = hyper_params["learning_rate"]
    lmd = hyper_params["weight_decay"]
    updated_model = model

    # TODO: Update the weights of each layer in your model based on the calculated gradients
    for layer_index, layer in enumerate(model["layers"]):
        for layer_param_name in layer["params"].keys():
            if layer_param_name == "W":
                layer["params"]["W"] -= a*grads[layer_index]["W"] + 2*lmd*layer["params"]["W"]
            else:
                layer["params"][layer_param_name] -= a*grads[layer_index][layer_param_name]
       
    return updated_model