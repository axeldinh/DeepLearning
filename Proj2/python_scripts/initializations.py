from torch import empty
import math

########
# Here we define different initialization techniques and provide functions to apply them.
# We decided to only implement the normal version of He and Xavier initialization.

def calculate_gain(nonlinearity):
    """
    Returns the gain, which depends on the linearity function
    """
    
    if nonlinearity.__name__ == 'Tanh':
        gain = 5/3
        
    elif nonlinearity.__name__ == 'ReLU':
        gain = math.sqrt(2)
    
    else:
        gain = 1
        
    return gain
    
    
def calculate_fan_in_fan_out(layer):
    """
    Returns the number of in features and out features of the layer
    """
    
    if layer.__class__.__name__ == 'Linear':
        return (layer.in_features, layer.out_features)
        
    else:
        return None, None
        

def normal_init(tensor, mean, std):
    """
    Returns a tensor of the same size as the original tensor from a normal(0, std^2) distribution
    """
    
    return    tensor.normal_(mean, math.pow(std,2))
    
def xavier_init(tensor, gain, fan_in, fan_out):
    """
    Returns a tensor with the normal Xavier initialization (normal(0, gain*sqrt(2/(fan_in, fan_out))))
    """
    
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    
    return normal_init(tensor, 0, std)
    
def he_init(tensor, gain, fan_in):
    """
    Returns a tensor with the normal He initialization (normal(0, gain*sqrt(1/fan_in)))
    """
    
    std = gain / math.sqrt(fan_in)
    
    return normal_init(tensor, 0, std)