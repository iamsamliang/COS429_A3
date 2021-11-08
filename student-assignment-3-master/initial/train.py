import sys
sys.path += ['layers']
import numpy as np
from loss_crossentropy import loss_crossentropy
import matplotlib.pyplot as plt

######################################################
# Set use_pcode to True to use the provided pyc code
# for inference, calc_gradient, loss_crossentropy and update_weights
use_pcode = False

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

def train(model, input, label, test_data, test_labels, params, numIters):
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
            params["eps"] (for decreasing lr when loss plateaus)
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
    
    # test data size
    test_data_size = test_labels.shape[0]
    
    # Friction rho for momentum implementation in GD
    rho = params.get("friction_rho", 0.99)
    
    # eps
    eps = params.get("eps", 1e-5)

    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features like momentum.
    update_params = {"learning_rate": lr,
                     "weight_decay": wd }

    num_inputs = input.shape[-1]
    loss = np.zeros((numIters,)) # TODO TODO TODO seems problematic? Need to construct loss/accuracy arrays for plotting!
    num_layers = len(model["layers"])
    
    # velocity initialization for momentum
    v = []
    for layer in model["layers"]:
        v.append({layer_param_name: np.zeros(layer["params"][layer_param_name].shape) for layer_param_name in layer["params"].keys()})
    
    # keep track of the training loss and testing loss
    training_losses = []
    test_losses = []
            
    for i in range(numIters):
        print(f'Starting iteration {i}')
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
        training_losses.append(loss)
#         print("Before test_loss")
#         if i % (numIters // 10) == 0:
#             test_output, _ = inference(model, test_data)
#             test_loss, _ = loss_crossentropy(test_output, test_labels, {}, False)
#             test_losses.append(test_loss)
#             test_accuracy = np.count_nonzero(test_output==test_labels) / test_data_size
#         print("After test_loss")
        # np.count_nonzero(output==training_labels) counts how many predicted_labels match the real labels
        train_accuracy = np.count_nonzero(output==training_labels) / batch_size
        
        # stop training if we hit 50% accuracy
        if train_accuracy >= 0.5:
            break
            
        # decrease learning rate when loss plateaus
        if i > 1 and (training_loss[-2] - training_loss[-1]) / training_loss[-1] < eps:
            lr = lr/2
            update_params['learning_rate'] = lr
            
        print("Before calculating gradient")
        #   (4) Calculate gradients
        gradients = calc_gradient(model, training_batch, activations, dv_input)
        print("After calculating gradient")
        
        #   (5) Update the weights of the model
        # implementing momentum
        print("Before Momentum")
        for layer_index in num_layers:
            v[layer_index] = {layer_param_name: rho*v[layer_index][layer_param_name] + gradients[layer_index][layer_param_name] for layer_param_name in layer["params"].keys()}
        print("After Momentum")
        
        print("Before Updating Weights")
        model = update_weights(model, v, update_params)
        print("After Updating Weights")
        
        print(f'Ending iteration {i}')
        
        # Optionally,
        
        #   (1) Monitor the progress of training
        if (i % (numIters // 10) == 0):
            print(f"---------- Iteration {i} of {numIters} ----------\n")
            print(f"Training Accuracy: {train_accuracy}")
            print(f"Testing Accuracy: {test_accuracy}")
        
        #   (2) Save your learnt model, using ``np.savez(save_file, **model)``
        np.savez(save_file, **model)
        
    return model, loss