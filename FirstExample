IMPORTANT: There is no learning proses in this network.
Learning proses of neural network means selection of weight values ​​by a neural network (sometimes by 'random'). 
But in this case we give values of weights by ourselves. 

# based on ex. 2
import numpy as np

# input layer
firts_value = 1.0

second_value = 0.0

third_value = 1.0


def activation_function(x):
    if x >= 0.5:
        return 1
    elif x < 0.5:
        return 0


def predict(firts_value, second_value, third_value):
    inputs = np.array([firts_value, second_value, third_value])


    # weights that hidden neurons take:
    weights_input_to_hidden1 = [0.25, 0.25, 0]
    # remember - the first neuron of the hidden layer took exactly these values
    # and we set the values ​​ourselves and the same with the second one:

    weights_input_to_hidden2 = [0.5, -0.4, 0.9]
    
    # weights from hidden layer tо output 
    weights_input_to_hidden = np.array((weights_input_to_hidden1, weights_input_to_hidden2))

# Line weights_input_to_hidden = np.array((weights_input_to_hidden1, weights_input_to_hidden2))
# creates a two-dimensional array (matrix) weights_input_to_hidden, which represents the weights
# connecting neurons in the hidden layer to neurons in the output layer of your neural network.

# In our case:

# weights_input_to_hidden1 represents the weights associated with the first neuron in the hidden layer.
# weights_input_to_hidden2 represents the weights associated with the second neuron in the hidden layer.
# Both of these neurons take input from the inputs array you defined earlier.

# Create the weights_input_to_hidden array using
# np.array((weights_input_to_hidden1, weights_input_to_hidden2))
# combines the weights for both neurons into one matrix,
# where the rows of the matrix represent the neurons in the hidden layer,
# and the columns represent the weights associated with each of the neurons.
# This allows you to efficiently manage and calculate weighted sums for each neuron in the hidden layer
# when passing input data through weights.

    # weights from hidden layer to output
    weights_hidden_to_output = np.array([-1, 1])

    # actual implementation of the formula
    # x1 * w1 + x2 * w2 + x3 * w3
    # based on this inputs is an array containing
    # the sum of the input data, and weights_input_to_hidden
    # - array containing the sum of weights
    hidden_input = np.dot(weights_input_to_hidden, inputs)

    print("hidden_input: " + str(hidden_input))
# predict(firts_value, second_value, third_value)



    hidden_output = np.array([activation_function(x) for x in hidden_input])
    print("hidden_output: " + str(hidden_output))


    output = np.dot(weights_hidden_to_output, hidden_output)
    print("output: " + str(output))
    return activation_function(output) == 1

print("result is: " + str (predict(firts_value, second_value, third_value)))
