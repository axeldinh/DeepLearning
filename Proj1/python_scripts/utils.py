import torch # Only library we are allowed to use

# Little script with basic functions

def normalize(train_input, test_input):
    """
    Normalizes the data, first puts the data in a [0, 1] range then normalizes each set by the train mean and std mean.
    
    Parameters:
        -train_input, test_input (tensor, tensor): datasets
        
    Returns:
        -train_input, test_input (tensor, tensor): normalized datasets
    """
    
    train_input = train_input / 255.
    test_input = test_input / 255. # Put the data from [0., 255.] to [0., 1.]

    mean = train_input.mean()
    std = train_input.std()
    
    train_input = (train_input - mean) / std
    test_input = (test_input - mean) / std
    
    return train_input, test_input
    
def compute_number_parameters(model):
    """
    Computes the number of parameters in a model.
    
    Parameters:
        - model (Module): instance of nn.Module
        
    Returns:
        total_params (int): number of parameters in the model
    """
    
    total_params = 0
    for p in model.parameters():
        total_params += torch.numel(p)
        
    return total_params
