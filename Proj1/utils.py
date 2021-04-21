import torch # Only library we are allowed to use

class Metrics():
    """
    Little class containing the desired metrics, for now it can take:
        -train_losses
        -test_losses
        -accuracy
    Each of them consist in a list of lists containing the metrics at each epoch, e.g
    [ [loss_epoch1_training1, ..., loss_epoch25_training1], ..., [loss_epoch1_training10, ..., loss_epoch25_training10] ]
    """
    
    def __init__(self):
        
        self.train_losses = []
        self.test_losses = []
        self.accuracies = []


def train(model, criterion, optimizer, train_input, train_target, test_input, test_target, mini_batch_size, epochs = 25):
    """
    Trains a model for some epochs with a batch size of mini_batch_size
    Returns:
        -train_losses (list): list of the train losses per epoch
        -test_losses (list): list of the test losses per epoch
        -accuracies (list): list of the accuracies per epoch
    """
    model.train()
    
    train_losses = []
    test_losses = []
    accuracies = []
    
    for epoch in range(epochs):
        
        train_loss = 0
        test_loss = 0
        accuracy = 0
        
        for b in range(0, train_input.size(0), mini_batch_size):
            
            optimizer.zero_grad()
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            loss.backward()
            optimizer.step()
            
            train_loss += loss
            
        train_loss /= mini_batch_size
        accuracy, test_loss = evaluate(model, criterion, test_input, test_target, mini_batch_size)
        
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        accuracies.append(accuracy.item())
        
    return train_losses, test_losses, accuracies

def evaluate(model, criterion, test_input, test_target, mini_batch_size):
    """
    Evaluate the model on the test set
    Returns:
        -accuracy (torch.tensor): shape (1)
        -test_loss (torch.tensor): shape(1)
    """
    
    model.eval()
    predicted = 0
    loss = 0
    
    with torch.no_grad():
        for b in range(0, test_input.size(0), mini_batch_size):

            output = model(test_input.narrow(0, b, mini_batch_size))
            loss += criterion(output, test_target.narrow(0, b, mini_batch_size))
            _, predictions = torch.max(output, dim = -1)
            predicted += torch.sum(predictions == test_target.narrow(0, b, mini_batch_size))
        
    accuracy = predicted / test_input.size(0)
        
    return  accuracy, loss / mini_batch_size

def many_trains(Modelfun, criterion, optimizer, train_input, train_target, test_input, test_target, mini_batch_size, num_trains, epochs = 25):
    """
    Trains the model num_trains times, so that we can get standard deviations
    Parameters:
        Modelfun: function handle to create the model
        criterion: criterion to use (in the form criterion(input, target))
        optimizer: probably SGD

    Returns:
        metrics (Metrics): object of the Metrics class
    """
    
    metrics = Metrics()
    
    for n_train in range(num_trains):
        
        model = Modelfun()
        optimizer.param_groups = [{'params': [p for p in model.parameters()],
                                   'lr': optimizer.defaults['lr'],
                                   'momentum':optimizer.defaults['momentum'],
                                   'dampening': optimizer.defaults['dampening'],
                                   'weight_decay': optimizer.defaults['weight_decay'], 
                                   'nesterov': optimizer.defaults['nesterov']}]
        
        print(f"Training {n_train+1}: ", end = '')
        train_losses, test_losses, accuracies = train(model, criterion, optimizer, train_input, train_target,
                                                      test_input, test_target, mini_batch_size, epochs)
        
        metrics.train_losses.append(train_losses)
        metrics.test_losses.append(test_losses)
        metrics.accuracies.append(accuracies)
        print("Final accuracy: {:.2f}%, Best Accuracy: {:.2f}%".format(accuracies[-1]*100, max(accuracies)*100))
    
    return metrics

