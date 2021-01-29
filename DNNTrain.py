import os,sys,inspect
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import *


#detection machine
from sys import platform as _platform
if _platform == "win32":
    scriptPATH = os.path.abspath(inspect.getsourcefile(lambda:0)) # compatible interactive Python Shell
    scriptDIR  = os.path.dirname(scriptPATH)

#Loading datasets
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

### CONSTANTS ###
in_l = 64*64*3 #W*H*RGBChannels
layers_dims = [in_l, 20, 7, 5, 1] #  4-layer model
learning_rate = 0.0075
num_iterations = 2500

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, par={}, print_cost=False, retrain=False):
    np.random.seed(1)
    costs = []# keep track of cost

    #Start over in case of different NN or continue with previous parameters in case of identical NN
    eq = True
    if retrain and len(par) == (len(layers_dims)-1)*2:
        for i in range(len(layers_dims)-1):
            if par['W'+str(i+1)].shape[0] != layers_dims[i+1]:
                eq = False
                break
    else: eq = False   
    if eq: parameters = par
    else: parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def save(parameters):
    #Saves or overwrites parameters dictionary
    with open('parameters.pickle', 'wb') as handle:
        pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

def initwb():
    #Loads previous parameters if exist
    for filename in os.listdir(scriptDIR):
        if filename.endswith(".pickle"):
            with open('parameters.pickle', 'rb') as f:
                par = pickle.load(f)
            start = time.time()
            parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate, num_iterations, par, print_cost = True, retrain = True)
            save(parameters)
            print ("it took", time.time() - start, "seconds.")
            return
    start = time.time()
    parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate, num_iterations, print_cost = True)
    print ("it took", time.time() - start, "seconds.")
    save(parameters)
    return

def main():
    initwb()

if __name__ == "__main__":
    main()





