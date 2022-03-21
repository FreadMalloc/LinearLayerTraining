from operator import truth
from random import randrange
import sys
import numpy as np
import torch
import torch.nn as nn

############################ definitions #################################
I_in = 1920                 # input size
J_out = 4                  # output size
W_weigh = I_in * J_out   # weights

# .h filename
fileName = "rand_init.h"

#string initialization
IN_vec_string = "#define INPUTS_UINT8 {"
WEIGHTS_vec_string = "#define WEIGHTS_FLOAT32 {"
BIASES_vec_string = "#define BIASES_FLOAT32 {"
VERIFICATION_string = "#define VERIFICATION_FLOAT32 {"


########################## data generation ###############################

# (I_in,1) uint8_t inputs
inputs = np.random.randint(0, 256, I_in)
#print(inputs)

# (W_weigh,1) float32 weights
weights = np.random.normal(0.0, 0.1, W_weigh)

# (J_out,1) float32 biasess
biases = np.random.normal(0.0, 0.1, J_out)
print(inputs.size, weights.size, biases.size)


################### writing the DEFINES on fileName.h for pulp #######################

# saving the string for inputs
for idx, i in enumerate(inputs):
    IN_vec_string += str(i)
    
    if idx != I_in-1:
        IN_vec_string += ", "
    else:
        IN_vec_string += "}\n"
    
    if idx % 100 == 0:
        IN_vec_string += "\\\n"         # every 100 entries we append \ and go on a new line, cause lines that are too long are not compiled correctly

# saving the string for weights
for idx, j in enumerate(weights):
    WEIGHTS_vec_string += str(j)
    
    if idx != W_weigh-1:
        WEIGHTS_vec_string += ", "
    else:
        WEIGHTS_vec_string += "}\n"

    if idx % 100 == 0:
        WEIGHTS_vec_string += "\\\n" 

# saving the string for inputs
for idx, b in enumerate(biases):
    BIASES_vec_string += str(b)
    
    if idx != J_out-1:
        BIASES_vec_string += ", "
    else:
        BIASES_vec_string += "}\n"


################# validating with pytorch (1 foreward propagation) #####################

layer = nn.Linear(in_features=I_in, out_features=J_out, bias=True)

weights = np.reshape(weights, (J_out,I_in))
weights = torch.from_numpy(weights).type(torch.float)
print("W size: " + str(weights.size()) )

biases = torch.from_numpy(biases).type(torch.float)
print("B size: " + str(biases.size()) )

inputs = torch.from_numpy(inputs).type(torch.float)
print("I size: " + str(inputs.size()) )

layer.weight = nn.Parameter(weights)
layer.bias = nn.Parameter(biases)
inputs = nn.Parameter(inputs)

outputs = layer(inputs)     #sent to .h for verification in the cluster


outputs_numpy = outputs.detach().numpy()
for idx, b in enumerate(outputs_numpy):   #saving the validation vector
    VERIFICATION_string += str(b)
    
    if idx != J_out-1:
        VERIFICATION_string += ", "
    else:
        VERIFICATION_string += "}\n"


############################ VALIDATING THE BACKPROPAGATION ###########################

loss_function = torch.nn.MSELoss()
learning_rate = 0.00000001
true_model = torch.FloatTensor([0.0, 1.0, 0.0, 0.0])

loss = loss_function(outputs, true_model)
print(layer(inputs))
print(loss)

layer.zero_grad()

loss.backward()

with torch.no_grad():
    for param in layer.parameters():
        param -= learning_rate * param.grad

loss = loss_function(layer(inputs), true_model)
print(layer(inputs))
print(loss)

#printing everything on .h file
txt_file = open(fileName, 'w')
txt_file.write(IN_vec_string + WEIGHTS_vec_string + BIASES_vec_string + VERIFICATION_string)
txt_file.close()