from .metrics import *
from .utils import *
from dlc_practical_prologue import *
import torch

############################
# Scripts with functions for training
# Contains:
#   - train
#   - evaluate
#   - many_trains
#   - get_mean_std
#   - cross_validate

def train(model, optimizer, mini_batch_size, epochs = 25, device = 'cpu'):
    """
    Trains a model for some epochs with a batch size of mini_batch_size
    Parameters:
        - model (Model): instance from the Model class
        - optimizer: instance from torch.optim
        - mini_batch_size (int): size of the batches for training and evaluation
        - epochs (int): number of epochs to train
        - device (str): which devide to train on ('cuda' or 'cpu')
    Returns:
        -train_losses (list): list of the train losses per epoch
        -test_losses (list): list of the test losses per epoch
        -accuracies (list): list of the accuracies per epoch
    """
    model.train()
    model.to(device)

    train_losses = []
    test_losses = []
    accuracies = []

    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)
    train_input, test_input =  normalize(train_input, test_input)
    
    train_input = train_input.to(device)
    train_target = train_target.to(device)
    train_classes = train_classes.to(device)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    test_classes = test_classes.to(device)
    
    for epoch in range(epochs):
        
        train_loss = 0
        test_loss = 0
        accuracy = 0
        
        for b in range(0, train_input.size(0), mini_batch_size):
            
            
            outputs = model(train_input.narrow(0, b, mini_batch_size))
            loss = model.loss(outputs, train_target.narrow(0, b, mini_batch_size), train_classes.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss
            
        train_loss /= mini_batch_size
        accuracy, test_loss = evaluate(model, test_input, test_target, test_classes, mini_batch_size)
        
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        accuracies.append(accuracy.item())
        
    return train_losses, test_losses, accuracies

def evaluate(model, test_input, test_target, test_classes, mini_batch_size):
    """
    Evaluate the model on the test set
    Parameters:
        - model (Model): instance from the Model class
        - test_input (Tensor): input for model.forward(), size = (*, 2, 14, 14)
        - test_target (Tensor): binary value of the digit comparison, size = (*, 1)
        - test_classes (Tensor): value of the digits in test_input, size = (*, 2)
        - mini_batch_size (int): size of the batches for training and evaluation

    Returns:
        -accuracy (torch.tensor): shape (1)
        -test_loss (torch.tensor): shape(1)
    """
    
    model.eval()
    predicted = 0
    loss = 0
    
    with torch.no_grad():
        for b in range(0, test_input.size(0), mini_batch_size):
            
            if b + mini_batch_size > test_input.size(0):
                b = test_input.size(0) - mini_batch_size

            outputs = model(test_input.narrow(0, b, mini_batch_size))
            loss += model.loss(outputs, test_target.narrow(0, b, mini_batch_size), test_classes.narrow(0, b, mini_batch_size))
            
            predicted += model.correct_predictions(outputs, test_target.narrow(0, b, mini_batch_size), test_classes.narrow(0, b, mini_batch_size) )[0]
        
    accuracy = predicted / test_input.size(0)
        
    return  accuracy, loss / mini_batch_size
    
def many_trains(Model_fn, config, optimizer, mini_batch_size, num_trains, epochs = 25, device = 'cpu'):
    """
    Trains the model num_trains times, so that we can get standard deviations
    Parameters:
        - Model_fn: function handle to create the model
        - criterion: criterion to use (in the form criterion(input, target))
        - optimizer: instance from torch.optim

    Returns:
        - metrics (Metrics): object of the Metrics class
    """
    
    metrics = Metrics()
    
    for n_train in range(num_trains):
        
        # Reinitialize the model and set the optimizer's parameters to the model parameters keeping everything else the same
        model = Model_fn(*config)
        optimizer.param_groups[0]['params']  = [p for p in model.parameters()]
        
        print(f"Training [{n_train+1}/{num_trains}]: ", end = '')
        train_losses, test_losses, accuracies = train(model, optimizer,
                                                      mini_batch_size, epochs, device)
        
        metrics.train_losses.append(train_losses)
        metrics.test_losses.append(test_losses)
        metrics.accuracies.append(accuracies)
        print("Final accuracy: {:.2f}%, Best Accuracy: {:.2f}%".format(accuracies[-1]*100, max(accuracies)*100))
    
    return metrics
    

def get_mean_std(Model_fn, config, optimizer, mini_batch_size, num_trains, epochs = 25, device = 'cpu'):
    """
    Returns the mean and the standard deviation of the best test accuracy achieved on num_trains trainings
    
    Parameters:
        - Model_fn: function handle to the model to train
        - config (list): list of arguments to pass to Model_fn (as Model_fn(*config))
        - optimizer: optimizer from the torch.optim module
        - mini_batch_size (int): batch size for training and evaluating
        - num_trains (int): number of trainings
        - epochs (int): number of epochs per training
        - device (str): which device to train on (either 'cuda' or 'cpu')
    """
    
    metrics = many_trains(Model_fn, config, optimizer, mini_batch_size = mini_batch_size, num_trains = num_trains, epochs = epochs, device = device)
    accuracies = torch.Tensor(metrics.accuracies)
    
    mean = accuracies.max(dim = 1).values.mean()
    std = accuracies.max(dim = 1).values.std()
    
    return mean, std, metrics
    
def cross_validate(Model_fn, configs, optimizers, k_folds = 5, mini_batch_size = 10, epochs = 25, device = 'cpu'):
    """
    Makes a cross validation and returns the best model out of the models defined by configs (Model_fn(*config[i])).
    The runs on each model are made on the same train set which is divided in k_folds folds.
    Each model are trained on the training folds then are evaluated on the validation folds, we then compare
    the models using the mean of the validation accuracies
    
    Parameters:
        - Model_fn (Model): function handle to a Model
        - configs (list(list)): list of configurations necessary to call Model_fn(*config)
        - optimizers (list(optimizer)): list of torch.optim instances, this way we can train on different learning rates
        - k_folds (int): number of times the training set is split for the cross validation
        - mini_batch_size (int): batch size for the training and evaluation
        - epochs (int): number of epochs per run
        - device (str): which device to train on ('cuda' or 'cpu')
        
    Returns:
        - best_config (list): configuration for the call to the best model Model_fn(*best_config).
        - best_model_accuracy (float): mean final accuracies obtained with the best model.
        - best_lr (float): learning rate used to obtained the best model
    """

    # Generate the data, it will be the same for all models, to avoid luck on the dataset generation
    full_train_input, full_train_target, full_train_classes, _, _, _ = generate_pair_sets(1000)
    full_train_input, _ = normalize(full_train_input, full_train_input)
    
    # Get the indices for the train set and validation set for each fold
    N = len(full_train_input)

    train_ratio = 1 - 1/k_folds
    train_samples = int(N * train_ratio)
    val_samples = N - train_samples
    rand_idx = torch.randperm(N).tolist()

    val_idx = [rand_idx[i*val_samples:(i+1)*val_samples] if (i+1)*val_samples <= N else rand_idx[i*val_samples:-1] for i in range(k_folds)]
    train_idx = [ [idx for idx in range(N) if idx not in val_idx[i]] for i in range(len(val_idx))]

    # Train each model
    mean_accuracies = []

    for i in range(len(configs)):

        print(f"Training {Model_fn.__name__} with config {configs[i]} and lr {optimizers[i].param_groups[0]['lr']} [{i+1}/{len(configs)}]:\n")

        accuracies = []

        for fold in range(k_folds):
            
            # Reinitialize the model and the optimizer

            model = Model_fn(*configs[i]).to(device)
            optimizer = optimizers[i]
            optimizer.param_groups[0]['params']  = [p for p in model.parameters()]

            print(f"Fold [{fold+1}/{k_folds}]: ", end = '')

            # Create the sets and train

            train_input = full_train_input[train_idx[fold]].to(device)
            train_target = full_train_target[train_idx[fold]].to(device)
            train_classes = full_train_classes[train_idx[fold]].to(device)
            val_input = full_train_input[val_idx[fold]].to(device)
            val_target = full_train_target[val_idx[fold]].to(device)
            val_classes = full_train_classes[val_idx[fold]].to(device)

            for epoch in range(epochs):
            
                for b in range(0, train_input.size(0), mini_batch_size):

                    if b + mini_batch_size > train_input.size(0):
                        b = train_input.size(0) - mini_batch_size
                    
                    optimizer.zero_grad()
                    outputs = model(train_input.narrow(0, b, mini_batch_size))
                    loss = model.loss(outputs, train_target.narrow(0, b, mini_batch_size), train_classes.narrow(0, b, mini_batch_size))
                    loss.backward()
                    optimizer.step()

            # Get the final validation accuracy
            accuracy, _ = evaluate(model, val_input, val_target, val_classes, mini_batch_size)
            accuracies.append(accuracy.item())

            print("Final Accuracy = {:.2f}%".format(accuracy*100))

        # Get the mean of accuracies over folds
        mean_accuracies.append(sum(accuracies) / k_folds)

        print("Mean Final Accuracies over folds = {:.2f}%\n".format(mean_accuracies[-1]*100))

    # return the best config with the corresponding accuracy and lr
    best_config_idx = torch.Tensor(mean_accuracies).max(dim = 0).indices.item()
    best_config = configs[best_config_idx]
    best_model_accuracy = torch.Tensor(mean_accuracies).max(dim = 0).values.item()
    best_lr = optimizers[best_config_idx].param_groups[0]['lr']

    print(f"Best config: {configs[best_config_idx]} with lr={best_lr}, Best Accuracy: {best_model_accuracy}")

    return best_config, best_model_accuracy, best_lr