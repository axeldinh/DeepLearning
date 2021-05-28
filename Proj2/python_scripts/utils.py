import math
from torch import empty
from .modules import Sequential, Linear, Dropout
from .initializations import *

def create_model(activation, out_features, p = 0.0, init_method = None):
    """
    Creates a model as we would like it to be:
    
    Parameters:
        -activation (function handle): activation function,
        -p (float): probability for dropout, set to 0.0 to avoid dropout
        -init_method (string): initialization method (either 'xavier', 'x', 'he')
    
    Returns:
        - model (Module): Model with the following structure
                          3*(Linear(2, 25) -> Dropout(p) -> activation) -> Linear(25, out_features)
                          
    """
    
    if (out_features != 1) and (out_features != 2):
        raise ValueError("out_features should be 1 or 2 but got {}".format(out_features))
    
    model = Sequential(
    [
        Linear(2,25),
        Dropout(p),
        activation(),
        Linear(25,25),
        Dropout(p),
        activation(),
        Linear(25,25),
        Dropout(p),
        activation(),
        Linear(25,out_features)
    ])
    
    if init_method is not None:
        gain = calculate_gain(activation)
        model.init_parameters(init_method, gain)
    
    return model

def make_dataset(num_samples = 1000):
    """
    Makes the Dataset, the inputs are sampled from 2 uniform distribution in [0,1].
    Sample inside a circle of center (0.5, 0.5) and radius 1/sqrt(2*pi) are labeled 1
    
    Parameters:
        - num_samples (int): number of samples in the dataset
    
    Returns:
        - inputs (tensor): tensor of size (num_samples, 2)
        - labels (tensor): tensor of size (num_sample, 1)
    
    """
    
    x = empty((num_samples, 2)).uniform_()
    dist = x.sub(0.5).pow(2).sum(axis = 1).sqrt()
    labels = (dist < 1 / math.sqrt(2*math.pi)).float()
    
    return x.float(), labels