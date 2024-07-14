
# We didn’t train the previous neural network, but we will train this 
# its going to be a perceptron
# the perceptron learns the dependencies between training_inputs and training_outputs

import numpy as np 

# using sigmoid as an activatio function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# creating input layer as an array 
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# create training output
# that's right - this is all the output column



# this is the desired result that we ultimately want to get
# in this way we create a dependency for the network between training_inputs and training_outputs
training_outputs = np.array([[1,1,1,0]]).T

# training_outputs = np.array([[0,1,1,0]]).T: An array is created here
# training_outputs, which is a matrix
# with one row and four columns.
# The array T denotes the matrix transpose,
# that is, rows become columns,
# and the output is a matrix with one column
# and four lines.
# So training_outputs represents
# vertical column with values ​​[0, 1, 1, 0].



np.random.seed(1)
synaptic_weights = 2 * np.random.random((1, 3)) - 1 

# np.random.random((1, 3)): This part of the code generates random numbers in the range [0, 1].
#(1, 3) is a tuple that indicates that we want to create a 1x3 array (one row and three columns).
# So np.random.random((1, 3)) creates a 1x3 array of random numbers.

# 2 * np.random.random((1, 3)): Here, each random number from the previous array is multiplied by 2.
# This causes the value range to change from [0, 1] to [0, 2]. The array now contains random numbers in the range [0, 2].

# - 1: After multiplying by 2, we subtract 1 to shift the range of values ​​to the interval [-1, 1]. - the most important
# since the value of the weights is set in this range
# The synaptic_weights array will now contain random weights ranging from -1 to 1.


# So the correct version of the line creates a 1x3 array of synaptic_weights with random starting weights,
# which can be used in a neural network to initialize connections between neurons.

# print("Random generation of weights:" +str (synaptic_weights))


#neuros training
# there are several methods: back propagation (main), resilet propagation, genetic algorithm and etc.
# we use back propagation
# this method involves repeated training
# in this case 9000 times

for i in range(9000):
# implementation of linear regression formula
    input_layer = training_inputs
  # the result that the network predicts before training
    output_layer = sigmoid(np.dot(input_layer, synaptic_weights.T))




# This line of code calculates the error
    # (difference between desired and actual results)
    # by subtracting the actual result from the desired result.
    # err becomes an error vector,
    # where each vector element represents the difference between
    # corresponding desired and actual output values.
    err = training_outputs - output_layer



  # This part of the code performs element-wise multiplication of the error vector
    # err on expression (output_layer * (1 - output_layer)).
    # This expression is the product of three vectors:
    # output_layer (actual output of the neural network), (1 - output_layer)
    # (values ​​that are derivatives of the activation function)
    # and err (error vector). This is an important expression
    # since it is used to calculate the weight correction which
    # based on error.


    adjustments = np.dot(input_layer.T, err * (output_layer * (1-output_layer)) )

    # synaptic_weights += adjustments.T: This line updates the weights
    # networks by adding adjustments to the current weights.
    # This is a step that leads to improvements in network parameters at each training iteration.


    synaptic_weights += adjustments.T

print("Weights after learning")
print(synaptic_weights)
print("Result (after learing):")
print(output_layer)




# test
new_inputs = np.array([1, 1, 0])
output_layer = sigmoid(np.dot (new_inputs, synaptic_weights.T))


print("Тест: ")
print(output_layer)
