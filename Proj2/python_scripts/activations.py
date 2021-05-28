from torch import empty
import math
from .modules import Module

#### ACTIVATIONS ####################
### Here we define 2 activation functions: Tanh and ReLU

class Tanh(Module):
    """
    Applies the hyperbolic tangent unit function element-wise
    """

    def __init__(self):
        super(Tanh, self).__init__()
        
        self.last_input = None
    
    def forward(self, x):
        self.last_input = x
        return x.tanh()
    
    def backward(self, gradwrtoutput = None):
        
        if gradwrtoutput == None:
            gradwrtoutput = empty(self.last_input.size()).fill_(1.)

        derivative = 1 - self.forward(self.last_input).pow(2)
        
        return derivative.mul(gradwrtoutput)   # Apply chainrule 
    
    def param(self):
        return []

class ReLU(Module):
    """
    Applies the rectified linear unit function element-wise
    """
    
    def __init__(self):
        super(ReLU, self).__init__()
        
        self.last_input = None
    
    def forward(self, x):
        self.last_input = x
        return x.relu()
    
    def backward(self, gradwrtoutput=None):
        
        if gradwrtoutput == None:
            gradwrtoutput = empty(self.last_input.size()).fill_(1.)
            
        derivative = (self.last_input > 0).float()
            
        return derivative.mul(gradwrtoutput)
    
    def param(self):
        return []