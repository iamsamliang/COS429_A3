import sys
sys.path += ['layers']
import numpy as np
from loss_crossentropy import loss_crossentropy

######################################################
# Set use_pcode to True to use the provided pyc code
# for inference, calc_gradient, loss_crossentropy and update_weights
use_pcode = True

# You can modify the imports of this section to indicate
# whether to use the provided pyc or your own code for each of the four functions.
if use_pcode:
    # import the provided pyc implementation
    sys.path += ['pyc_code']
    from inference_ import inference
    from calc_gradient_ import calc_gradient
    from update_weights_ import update_weights
else:
    # import your own implementation
    from inference import inference
    from calc_gradient import calc_gradient
    from update_weights import update_weights
######################################################

def train(model, input, label, params, numIters):
    '''
    This training function is written specifically for classification,
    since it uses crossentropy loss and tests accuracy assuming the final output
    layer is a softmax layer. These can be changed for more general use.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [num_inputs]
        label: [num_inputs]
        params: Paramters for configuring training
            params["learning_rate"]
            params["weight_decay"]
            params["batch_size"]
            params["save_file"]
            params["friction_rho"] (ADDED FOR MOMENTUM)
            Free to add more parameters to this dictionary for your convenience of training.
        numIters: Number of training iterations
    '''
    # Initialize training parameters
    # Learning rate
    lr = params.get("learning_rate", .01)
    # Weight decay
    wd = params.get("weight_decay", .0005)
    # Batch size
    batch_size = params.get("batch_size", 128)
    # There is a good chance you will want to save your network model during/after
    # training. It is up to you where you save and how often you choose to back up
    # your model. By default the code saves the model in 'model.npz'.
    save_file = params.get("save_file", 'model.npz')
    
    # Friction rho for momentum implementation in GD
    rho = params.get("friction_rho", 0.99)

    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features like momentum.
    update_params = {"learning_rate": lr,
                     "weight_decay": wd }

    num_inputs = input.shape[-1]
    loss = np.zeros((numIters,)) # TODO TODO TODO seems problematic? Need to construct loss/accuracy arrays for plotting!
    num_layers = len(model["layers"])
    
    # velocity initialization for momentum
    v = []
    for layer_index, layer in enumerate(model["layers"]):
        v[layer_index] = {layer_param_name: np.zeros(layer["params"][layer_param_name].shape) for layer_param_name in layer["params"].keys()}
    
            
    for i in range(numIters):
        # TODO: One training iteration
        # Steps:
        #   (1) Select a subset of the input to use as a batch
            # generate batch_size number of random unique(replace=False) indices from the range of 0 - num_inputs (inclusive)
        ran_indices = np.random.choice(num_inputs, size=batch_size, replace=False)
        training_batch = input[..., ran_indices]
        training_labels = label[ran_indices]
        
        #   (2) Run inference on the batch
        output, activations = inference(model, training_batch)
        
        #   (3) Calculate loss and determine accuracy
            # dv_input = derivative of the loss with respect to the input
        loss, dv_input = loss_crossentropy(output, training_labels, {}, True)
        # np.count_nonzero(output==training_labels) counts how many predicted_labels match the real labels
        accuracy = np.count_nonzero(output==training_labels) / batch_size
        
        #   (4) Calculate gradients
        gradients = calc_gradient(model, training_batch, activations, dv_input)
        
        #   (5) Update the weights of the model
        # implementing momentum
        for layer_index in num_layers:
            v[layer_index] = {layer_param_name: rho*v[layer_index][layer_param_name] + gradients[layer_index][layer_param_name] for layer_param_name in layer["params"].keys()}
            
        model = update_weights(model, v, params)
        
        # Optionally,
        
        #   (1) Monitor the progress of training
        if (i % (numIters // 10) == 0):
            print(f"---------- Iteration {i} of {numIters} ----------\n")
            print(f"Training Accuracy: {accuracy}")
            print(f"Testing Accuracy: TODOTOTODOTODOTO")
        
        #   (2) Save your learnt model, using ``np.savez(save_file, **model)``
        np.savez(save_file, **model)
        
    return model, loss