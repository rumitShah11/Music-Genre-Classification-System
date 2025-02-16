import numpy as np
from random import random


# save activations and derivatives 
# implemen back propagation
# implement gradient descent
# implement train    
# train our net with osme dummy dataset
# make some prediction

class MLP(object): # multi layered perceptron

    def __init__(self, num_inputs=3, hidden_layers=[3,3], num_outputs=2):

        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs
        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        #inititate random weights for layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        derivatives=[]
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)           #append a in activations (list of arrays) where each array in the list will represent activation for a given layer 
        self.derivatives = derivatives

        activations=[]
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)           #append a in activations (list of arrays) where each array in the list will represent activation for a given layer 
        self.activations = activations

        
    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.
        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropogation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):

            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # save the activations for backpropogation
            self.activations[i+1] = activations

        # a(3) = sig(h(3))
        # h(3) = a(2)*W(2)

        # return output layer activation
        return activations 


    def back_propagate(self, error, verbose = False):
        """Backpropogates an error signal.
        Args:
            error (ndarray): The error to backprop.
        Returns:
            error (ndarray): The final error of the input
        """

        # dE/dW (i) = (y - a[i+1]) s'(h[i+1])) a(i)
        # s'(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1]

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = error * self._sigmoid_derivative(activations)

            # reshape delta as to have it as a 2d array
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i] 

            # reshape activations as to have them as a 2d column matrix
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1) #restructure the activations in a 2D form
            
            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations_reshaped,delta_reshaped)
            
            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i])) 

        return error


    def gradient_descent(self, learningRate=1):
        """Learns by descending the gradient
        Args:
            learningRate (float): How fast to learn.
        """
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            # print("Original W{}={}".format(i, weights))
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate
            # print("Updated W{}={}".format(i, weights))


    def train(self, inputs, targets, epochs, learning_rate):
        """Trains model running forward prop and backprop
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """
        # now enter the training loop epochs->number of times u wanna feed dataset to NN
        for i in range(epochs):
            sum_errors = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network
                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                # now perform gradient descent on the derivatives
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(inputs), i+1))

        print("Training complete!")
        print("=====")



    def _mse(self, target, output):
        """Mean Squared Error loss function
        Args:
            target (ndarray): The ground trut
            output (ndarray): The predicted values
        Returns:
            (float): Output
        """
        return np.average((target - output) ** 2)


    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

if __name__ == '__main__':

    #create a dataset to train a network for sum operation

    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])
    #create MLP

    mlp = MLP(2, [5], 1)

    #train the MLP

    mlp.train(inputs, targets, 500, 0.1)

    #create dummy data              
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    output = mlp.forward_propagate(input)
    print()
    print()
    print()

    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))
    #create inputs

    #inputs = np.array([0.1, 0.2])
    #target = np.array([0.3])

    #perform forward_propagate

    #output = mlp.forward_propagate(inputs)

    # calculate error
    #error = target - output

    #perform back_propagate

    #mlp.back_propagate(error)

    #apply gradient descent

    #mlp.gradient_descent(learningRate=0.1)

    # print("Network input is : {}".format(inputs))
    
    # print("Network output is : {}".format(outputs))



