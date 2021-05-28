from torch import empty
from .initializations import *
import math

#####################################
###### MODULES ######################

# The module class, from which we will define the classes necessary for the back-propagation
class Module(object):
    """
    Class from which all module will inherit
    
    Attributes:
        - training (bool): defines if the Module is in training mode
    
    Functions:
        - forward(self, *input)
        - backward(self, *gradwrtouput)
        - param()
        - zero_grad()
        - train()
        - eval()
        - init_parameters(, method, gain)
    """

    def __init__(self):

        self.training = True
    
    def forward(self, *input):
        """
        Returns the output of the Module
        """
        raise NotImplementedError
        
    def backward(self, *gradwrtouput):
        """
        Returns the gradient with respect to the last input.
        Takes into account previously computed gradient
        """
        raise NotImplementedError
        
    def param(self):
        """
        Returns a list containing tuples of the form (parameter, gradient)
        """
        return []
    
    def zero_grad(self):
        """
        Sets all parameters' gradients to 0.
        """
        for p, g in self.param():
            g.fill_(0.)

    def train(self):
        """
        Sets the training attribute to True (necessary to control the behaviour of the model)
        """
        self.training = True

    def eval(self):
        """
        Set the training attribute to False (necessary to control the behaviour of the model)
        """
        self.training = False

    def init_parameters(self, method = 'default', gain = 1.):
        """
        Initialize parameters for three different methods: xavier, and he.
        
        Parameters : 
              - method (string): initialization method to be applied. Accepted 
                                 values are ['xavier', 'he', 'default]
              - gain (float)   : gain defined by the nonlinearity function, get be found by calling calculate_gain(nonlinearity)
        """
        assert method in ['xavier', 'he', 'default'],\
            'given weight initialization method {} is not implemented'.format(method)
            
        fan_in, fan_out = calculate_fan_in_fan_out(self)
        
        if (fan_in == None) and (fan_out == None):
            return
            
        for w, _ in self.param():
            
            if method == 'xavier':
                xavier_init(w, gain, fan_in, fan_out)
                
            elif method == 'he':
                he_init(w, gain, fan_in)
            
            elif method == 'default':
                a = 1 / math.sqrt(fan_in)
                w.uniform_(-a,a)

    
##### LAYERS #########################
### Layers defined: Linear, Dropout, Sequential

class Linear(Module):
    """
    Class that implements a linear layer from in_features to out_features

    Parameters: 
          - in_features (in): number of input nodes
          - out_features (int): number of output nodes
          - weights (tensor[out_features x in_features]): weights
          - bias (tensor[out_features]): bias
          - grad_weigths (tensor[out_features x in_features]): store gradients of weights to cumulate
          - grad_bias (tensor[out_features]): store gradients of biases to cumulate
          - last_input (tensor[B x in_features]): Record of the last input for later back-propagation where B is the batch size
    """
    
    def __init__(self, in_features, out_features):
        
        super(Linear, self).__init__()

        if (in_features <= 0) or (out_features <= 0):
          raise ValueError("Both in_features and out_features should be > 0 but got:\n\
          in_features = {}, out_features = {}".format(in_features, out_features))
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize the weights and bias with a uniform(-a, a) distribution (as pytorch does)
        a = 1/math.sqrt(self.in_features)
        self.weights = empty((out_features, in_features)).uniform_(-a, a)
        self.bias = empty((out_features)).uniform_(-a,a)
        
        # Initialize gradients at 0.0
        self.grad_weights = empty((out_features, in_features)).fill_(0.)
        self.grad_bias = empty((out_features)).fill_(0.)
        
        self.last_input = None
        
        
    def forward(self, x):
        
        self.last_input = x
        output = self.weights.unsqueeze(0).matmul(self.last_input.unsqueeze(-1)).squeeze(-1)
        output = output.add_(self.bias.unsqueeze(0))
        
        return output
    
    def backward(self, gradwrtoutput = None):
        
        if gradwrtoutput == None:
            gradwrtoutput = empty(self.last_input.size()).fill_(1.)

        x = self.last_input
        self.grad_weights += gradwrtoutput.view(-1, self.out_features, 1).matmul(x.view(-1, 1, self.in_features)).mean(axis=0)
        self.grad_bias += gradwrtoutput.mean(axis = 0)
        
        gradwrtinput = gradwrtoutput.matmul(self.weights)
        
        return gradwrtinput.view(x.size())
        
        
    def param(self):
        return [(self.weights, self.grad_weights), (self.bias, self.grad_bias)]


class Dropout(Module):
    """
    Class Dropout: Takes a tensor as input and returns the same tensor, but replacing
    some elements by 0 with probability p. Also multiplies the output by 1/(1-p).
    
    Attributes:
        - p (float): probability to remove an output (should be between 0 and 1)
        - last_input (tensor): last tensor passed into forward()
        - mask (tensor): tensor to remember what ipnut have been zeroed
    """

    def __init__(self, p):

        if (p < 0) or (p >= 1):
            raise ValueError("p should be between 0 and 1 (1 excluded) but got p = {}".format(p))

        self.p = p
        self.last_input = None
        self.mask = None

    def forward(self, x):
        if self.training:
            self.last_input = x
            # get a mask of 0 and 1 and multiply the input by it
            self.mask = empty(x.size()).bernoulli_(p = 1-self.p) / (1-self.p)
            return (self.last_input * self.mask)
        else:
            return x

    def backward(self, gradwrtoutput):

        return self.mask * gradwrtoutput
    
    
class Sequential(Module):
    """
    A sequential container of modules. Modules will be iterated over in the order 
    they are passed in the constructor.

    Attributes: 
          - modules (list(Module)): list of modules to use one after another
    """
    def __init__(self, modules):
        
        super(Sequential, self).__init__()
        
        self.modules = modules
        
        
    def forward(self, x):
        # Iterate over modules in list order
        for module in self.modules:
            x = module.forward(x)
        return x
    
    def backward(self, gradwrtoutput):
        # Iterate over modules in inversed list order
        for module in self.modules[::-1]:
            new_grad = module.backward(gradwrtoutput)
            gradwrtoutput = new_grad
            
        return gradwrtoutput
    
    def param(self):
        # Put the modules' parameters into one list
        parameters = [p for module in self.modules for p in module.param()]
        return parameters

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()
            
    def init_parameters(self, method, gain):
        """
        Initialize parameters for three different methods: xavier, he and 'default'.
        
        parameters : 
              - method (string): initialization method to be applied. Accepted 
                      values are ['xavier', 'he', 'default']
              - gain (float): gain defined by the nonlinearity function, get be found by calling calculate_gain(nonlinearity)
        """
        assert method in ['xavier', 'he', 'default'],\
            'given weight initialization method {} is not implemented'.format(method)
            
        for module in self.modules:
            module.init_parameters(method, gain)