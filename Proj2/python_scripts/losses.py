from torch import empty
import math
from .modules import Module

##### LOSSES ###########################
### Here we define 2 losses: LossMSE and CrossEntropyLoss

class LossMSE(Module):
    """
    Measures the mean squared error between an output and its true labels.

    Attributes:
          - last_input (tensor): Record of the last input for later back-propagation
          - last_labels (tensor): true labels of last_input
    """
    def __init__(self):
        
        super(LossMSE, self).__init__()   
        
        self.last_input = None
        self.last_labels = None
        
    def forward(self, output, labels):
        
        self.last_input = output
        self.last_labels = labels
        
        return output.view(-1).sub(labels.view(-1)).pow(2).mean() 
    
    def backward(self, gradwrtoutput=None):
        
        if gradwrtoutput == None:
            gradwrtoutput = empty(self.last_input.size()).fill_(1)
        
        grad = (self.last_input.view(self.last_input.size(0), -1) - self.last_labels.view(self.last_input.size(0), -1))
        
        return 2 * grad.view(self.last_input.size()).mul(gradwrtoutput)
    
class CrossEntropyLoss(Module):
    """
    Measures the cross entropy between an output and it's true labels.

    Attributes:
          - last_input (tensor): Record of the last input for later back-propagation
          - last_labels (tensor): true labels of last_input
    """
    def __init__(self):
        
        super(CrossEntropyLoss, self).__init__()
        
        self.last_input = None
        self.last_labels = None
        
        
    def forward(self, outputs, labels):
        
        self.last_input = outputs
        self.last_labels = labels
        
        entropies = empty(outputs.size(0))
        for i in range(outputs.size(0)):
            entropies[i] = -(outputs[i,labels[i].long()].exp().div(outputs[i,:].exp().sum())).log()
            
        return entropies.mean()
    
    def backward(self, gradwrtoutput = None):
        
        if gradwrtoutput == None:
            gradwrtoutput = empty(self.last_input.size()).fill_(1.)
            
        derivatives = empty(self.last_input.size())
        
        for i in range(derivatives.size(0)):
            exp_sum = self.last_input[i].exp().sum()
            derivatives[i] = self.last_input[i].exp().div(exp_sum)
            derivatives[i, self.last_labels[i].long()] -= 1
            
        gradwrtinput = derivatives * gradwrtoutput
        
        return gradwrtinput.view(self.last_input.size()) / self.last_input.size(0)