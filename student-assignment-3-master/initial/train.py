import sys
sys.path += ['layers']
import numpy as np
from loss_crossentropy import loss_crossentropy
import matplotlib.pyplot as plt
import pickle

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
            params["save_metrics_file"]
            params["metrics"]: a dictionary of three keys
                training_metrics is a dictionary, which includes three keys:
                    - 'training_recording_epochs': 1D array, the (floating point) epoch values when the losses
                    and accuracies with the same index in the following two arrays were calculated.
                    - 'training_losses': 1D array of training losses, calculated over the training minibatch.
                    - 'training_accs': 1D array of testing losses, calculated over the testing minibatch.

                testing_metrics is similar to the training_metrics.

                curr_epoch is a floating number, used for tracking the process.
                
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
    save_metrics_file = params.get("save_metrics_file", "metrics.pickle")
    
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
    num_layers = len(model["layers"])
    
    # velocity initialization for momentum
    v = []
    for layer in model["layers"]:
        v.append({layer_param_name: np.zeros(layer["params"][layer_param_name].shape) for layer_param_name in layer["params"].keys()})
    
    
    test_batch_size = 1000
    test_report_freq = 10
    
    train_report_freq = 1
    
    # keep track of the training and testing metrics
    if params["metrics"] is None:
        params["metrics"] = {}
        params["metrics"]["curr_epoch"] = 0
        params["metrics"]["training_metrics"] = {'training_losses': [],
                                                 'training_accs': [],
                                                 'training_recording_epochs': []}
        params["metrics"]["testing_metrics"] = {'testing_losses': [],
                                                'testing_accs': [],
                                                'testing_recording_epochs': []}
    
    save_model_freq = 20

    for i in range(numIters):
        # TODO: One training iteration
        # Steps:
        
        ##########################################################
        #   (1) Select a subset of the input to use as a batch
        
        # generate batch_size number of random unique(replace=False) indices from the range of 0 - num_inputs (inclusive)
        ran_indices = np.random.choice(num_inputs, size=batch_size, replace=False)
        
        # overfitting.
        # ran_indices = range(batch_size)
        
        training_batch = input[..., ran_indices]
        training_labels = label[ran_indices]
        
        ##########################################################
        #   (2) Run inference on the batch
        output, activations = inference(model, training_batch)
        
        ##########################################################
        #   (3) Calculate loss and determine accuracy
        
        # for training
        loss, dv_input = loss_crossentropy(output, training_labels, {}, True)
        
        pred_train_labels = np.argmax(output, axis=0)
        
        # counts how many predicted_labels match the real labels
        train_accuracy = np.count_nonzero(pred_train_labels==training_labels) / batch_size
        
        if i % train_report_freq == 0:
            params["metrics"]["training_metrics"]['training_losses'].append(loss)
            params["metrics"]["training_metrics"]['training_accs'].append(train_accuracy)
            params["metrics"]["training_metrics"]['training_recording_epochs'].append(params["metrics"]["curr_epoch"])
         
        
        # for testing
        if i % test_report_freq == 0:
            
            test_batch_indices = np.random.choice(test_data.shape[-1], size=test_batch_size, replace=False)
            test_batch = test_data[...,test_batch_indices]
            test_batch_labels = test_labels[test_batch_indices]
            
            test_output, _ = inference(model, test_batch)
            pred_test_labels = np.argmax(test_output, axis=0)
            
            test_loss, _ = loss_crossentropy(test_output, test_batch_labels, {}, False)
            test_accuracy = np.count_nonzero(pred_test_labels==test_batch_labels) / test_batch_size
            params["metrics"]["testing_metrics"]['testing_losses'].append(test_loss)
            params["metrics"]["testing_metrics"]['testing_accs'].append(test_accuracy)
            params["metrics"]["testing_metrics"]['testing_recording_epochs'].append(params["metrics"]["curr_epoch"])
                                           
        # decrease learning rate when loss plateaus
        if i > 1 and (prev_loss - loss)/prev_loss < eps:
            lr /= 2
            update_params['learning_rate'] = lr
        prev_loss = loss
        
        ##########################################################
        #   (4) Calculate gradients
        gradients = calc_gradient(model, training_batch, activations, dv_input)
     
        ##########################################################
        #   (5) Update the weights of the model
        
        # implementing momentum
        for layer_index in range(num_layers):
            v[layer_index] = {layer_param_name: rho*v[layer_index][layer_param_name] + gradients[layer_index][layer_param_name] for layer_param_name in layer["params"].keys()}
     
        model = update_weights(model, v, update_params)
        
        ##########################################################
        #   (6) Monitor the progress of training
        print("---------- Iteration {i:d} of {iters:d} --- Epoch {curr_epoch:.4f} ----------\n".format(i=i, iters=numIters - 1, curr_epoch=params["metrics"]["curr_epoch"]))
        print(f"Training Accuracy: {train_accuracy}")
        print(f"Training Loss: {loss}")
        
        if i % test_report_freq == 0:
            print(f"Testing Accuracy: {test_accuracy}")
            print(f"Testing Loss: {test_loss}")
        
        print()

        ##########################################################
        #   (7) Keep track of the current epoch for visualization purposes
        #   noting that final iteration will be create the current epoch for the next
        #   run of the model.
        params["metrics"]["curr_epoch"] += batch_size/num_inputs
        
        ##########################################################
        #   (8) Save your learnt model, using ``np.savez(save_file, **model)``
        if i != 0 and i % save_model_freq == 0:
            np.savez(save_file, **model)
            with open(save_metrics_file, 'wb') as file:
                pickle.dump(params["metrics"], file, protocol=pickle.HIGHEST_PROTOCOL)
        
    return model