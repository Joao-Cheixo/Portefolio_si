import copy
from abc import abstractmethod

import numpy as np

from si.neural_networks.optimizers import Optimizer


class Layer:
    """
    Base class for neural network layers.
    """

    @abstractmethod
    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input, i.e., computes the output of a layer for a given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error: float) -> float:
        """
        Perform backward propagation on the given output error, i.e., computes dE/dX for a given dE/dY and update
        parameters if any.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        raise NotImplementedError

    def layer_name(self) -> str:
        """
        Returns the name of the layer.

        Returns
        -------
        str
            The name of the layer.
        """
        return self.__class__.__name__

    @abstractmethod
    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        raise NotImplementedError

    def set_input_shape(self, shape: tuple):
        """
        Sets the shape of the input to the layer.

        Parameters
        ----------
        shape: tuple
            The shape of the input to the layer.
        """
        self._input_shape = shape

    def input_shape(self) -> tuple:
        """
        Returns the shape of the input to the layer.

        Returns
        -------
        tuple
            The shape of the input to the layer.
        """
        return self._input_shape

    @abstractmethod
    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        raise NotImplementedError


class DenseLayer(Layer):
    """
    Dense layer of a neural network.
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        """
        Initialize the dense layer.

        Parameters
        ----------
        n_units: int
            The number of units of the layer, aka the number of neurons, aka the dimensionality of the output space.
        input_shape: tuple
            The shape of the input to the layer.
        """
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer: Optimizer) -> 'DenseLayer':
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> float:
        """
        Perform backward propagation on the given output error.
        Computes the dE/dW, dE/dB for a given output_error=dE/dY.
        Returns input_error=dE/dX to feed the previous layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        # computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        input_error = np.dot(output_error, self.weights.T)
        # computes the weight error: dE/dW = X.T * dE/dY
        weights_error = np.dot(self.input.T, output_error)
        # computes the bias error: dE/dB = dE/dY
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # updates parameters
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error

    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return (self.n_units,)
    



class DropoutLay(Layer):
    """Dropout layer of a neural network

    Args:
        probability (float): The probability of dropping out a neuron during training.

    Attributes:
        input (np.ndarray): The input to the dropout layer.
        mask (np.ndarray): The binary mask used for dropout.
        output (np.ndarray): The output of the dropout layer.

    """

    def __init__(self, probability: float):
        super().__init__()

        self.probability = probability

        self.input = None
        self.mask = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation through the dropout layer.

        Args:
            input (np.ndarray): The input to the dropout layer.
            training (bool): Whether the model is in training mode or not.

        Returns:
            np.ndarray: The output of the dropout layer.
        """
        

        if training:
            
            #compute the scaling factor
            scale_factor = 1 / (1 - self.probability)

            #generate the binary mask
            self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape)

            self.output = input * self.mask * scale_factor

            return self.output
        
        # if we are in inference mode, return the input

        else:
            self.output = input

            return self.output

    
    def backward_propagation(self, output_error: np.ndarray) -> float:
        """
        Performs backward propagation to update the weights of the layer.

        Args:
            output_error (np.ndarray): The error of the layer's output.

        Returns:
            float: The updated error of the layer's input.
        """
        # compute the input error
        input_error = output_error * self.mask

        return input_error
    

    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns:
            tuple: The shape of the output of the layer.
        """
        # the output shape is the same as the input shape
        return self.input_shape()
    
    def parameters(self) -> int:
            # the dropout layer has no parameters
            return 0
    
if __name__ == '__main__':
        #Testes para DenseLayer
        dense_layer = DenseLayer(n_units=5, input_shape=(10,))
        dense_layer.initialize(Optimizer(learning_rate=0.01))  

        print("Dense Layer - Output Shape:", dense_layer.output_shape())
        print("Dense Layer - Number of Parameters:", dense_layer.parameters())

        #criar um input de exemplo
        input_example = np.random.rand(100, 10)

        #Testa forward propagation na Dense Layer
        output_dense = dense_layer.forward_propagation(input_example, training=True)
        print("Dense Layer - Forward Propagation Output Shape:", output_dense.shape)

        #Testes para Dropout
        dropout_layer = DropoutLay(probability=0.5)

        #Teste forward propagation Dropout Layer
        output_dropout = dropout_layer.forward_propagation(output_dense, training=True)
        print("Dropout Layer - Forward Propagation Output Shape:", output_dropout.shape)

        # este backward propagation no Dropout Layer
        dropout_error_example = np.random.rand(100, 5)  #Exemplo de erro de saída
        dropout_input_error = dropout_layer.backward_propagation(dropout_error_example)
        print("Dropout Layer - Backward Propagation Input Error Shape:", dropout_input_error.shape)