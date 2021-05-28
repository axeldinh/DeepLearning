from .utils import *

def convert_to_predictions(output):
    """
    Converts an output into predictions
        If output.size() = (B, 1) then we return 1 if output > 0.5 (used for MSE)
        Else we return the argmax (useful for CrossEntropy)
        B is the batch size
        
    Parameters:
        - output (tensor[B, N]): tensor to convert
        
    Returns:
        - predictions (tensor[B, 1]): predictions given the output
    """
    
    if output.shape[1] < 2:
      return (output.view(-1) > 0.5).float()
    
    else:
      return output.argmax(dim = 1)
      

def multiple_training(model_config, criterion, optimizer, num_epochs = 200, batch_size = 10,
                      scheduler = None, num_trainings = 10, display = True):
                          
    """
    Trains the model num_trainings times.
    
    Parameters:
        - model_config (list): configuration necessary to call create_model(*model_config)
        - criterion (Module): Module used for the loss
        - optimizer (Optimizer): instance from the optimizer class, used for parameters update
        - num_epochs (int): number of epochs used for 1 training
        - batch_size (int): batch size used during training and evaluation
        - scheduler (Optimizer): instance from the Optimizer class, used for learning rate update
        - num_trainings (int): number of consecutive trainings (with reinitialization of the model and the data)
        - display (bool): whether or not we show the training steps
    
    Returns:
        - train_losses (list(list(int))): contains the train losses, train_losses[i][j] contains the train loss at training number i and epoch j
        - test_losses (list(list(int))): contains the test losses, test_losses[i][j] contains the test loss at training number i and epoch j
        - train_errors (list(list(int))): contains the train errors, train_errors[i][j] contains the train error rate at training number i and epoch j
        - test_errors (list(list(int))): contains the test errors, test_errors[i][j] contains the test error at training number i and epoch j
    """
    
    # Records of losses and errors
    train_losses  = []
    test_losses   = []
    train_errors  = []
    test_errors   = []
    
    # Keep the original learning rate for reinitialization
    lr = optimizer.lr
    
    for train_step in range(num_trainings):
        
        print(f"Training [{train_step+1}/{num_trainings}]: ", end = '')
        
        X_train, train_labels = make_dataset() # Make the datasets
        X_test, test_labels = make_dataset()
        
        model = create_model(*model_config) # Create the model
        optimizer.parameters = model.param() # Put the model's parameters into the optimizer
        
        # Reinitialize the scheduler and the learning rate
        if scheduler is not None:
            optimizer.lr = lr
            scheduler.optimizer = optimizer
            scheduler.init()
        
        # Train
        train_loss, test_loss, train_error, test_error = training(model, criterion, optimizer, X_train, X_test, train_labels, 
                                                               test_labels, num_epochs, batch_size, scheduler, display)
        
        # Store the metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        print("Best train error rate = {:.2f}%, Best test error rate = {:.2f}%".format(min(train_error)*100, min(test_error)*100))

    return train_losses, test_losses, train_errors, test_errors
    

def training(model, criterion, optimizer, X_train, X_test, train_labels,
             test_labels, num_epoch=200, batch_size=10, scheduler=None, display = True):
    """
    Trains the given model.

    Args:
          model: Module - Network to train on
          criterion: Module - loss function
          optimizer: Module - optimizer
          X_train: tensor - training data
          X_test: tensor - test data
          train_labels: tensor - train labels
          test_labels: tensor - test labels
          num_epochs: integer - number of epochs to train for
          batch_size: integer - number of samples per batch
          scheduler: Module - scheduler for optimal learning rate
    """

    # Records of losses and errors
    train_loss  = []
    test_loss   = []
    train_error = []
    test_error  = []

    for epoch in range(num_epoch):
      
        model.train()  # Switch model into training mode (important for dropouts)

        # Train one epoch and save train losses and errors
        tr_loss, tr_err = train(model, criterion, optimizer, X_train, train_labels, batch_size)
        train_loss.append(tr_loss.item())
        train_error.append(tr_err.item())

        model.eval()  # Switch model into evaluation mode (important for dropouts)

        # Evaluate loss and error over the test data
        te_loss, te_err = evaluate(model, criterion, X_test, test_labels, batch_size)
        test_loss.append(te_loss.item())
        test_error.append(te_err.item())

        # Optimize learning rate 
        if scheduler is not None:
          scheduler.step(te_loss)

        # Print values to look at something while waiting
        if ((epoch+1)%20 == 0) and display:
            print("Epoch [{}/{}]: Train Loss = {:.4f}, Test Loss = {:.4f}".format(epoch+1, num_epoch, train_loss[-1], test_loss[-1]))
            print("Train Error = {:.2f}%, Test Error = {:.2f}%".format(tr_err * 100, te_err*100))

    return train_loss, test_loss, train_error, test_error

def train(model, criterion, optimizer, X, labels, batch_size):
    """
    Train one epoch. Update parameters for each batch.

    Args:
         model: Module - Network to train on
          criterion: Module - loss function
          optimizer: Module - optimizer
          X: tensor - training data
          labels: tensor - train labels
          batch_size: integer - number of samples per batch

    Outputs:
          loss: integer - mean loss over all batches
         error: integer - mean error over all samples
    """
    loss  = 0
    error = 0

    for i in range(0, X.size(0), batch_size):
        
        # Select batch
        x = X[i:i+batch_size, :]
        lab = labels[i:i+batch_size] 

        # Forward 
        output = model.forward(x)
        loss  += criterion.forward(output, lab.long())

        # Compute error
        pred   = convert_to_predictions(output)   
        error += (pred != lab).sum()
        
        # Backward
        model.zero_grad()
        model.backward(criterion.backward())
        optimizer.step()

    # Average loss over batches and error over samples
    loss  = loss/(len(X)/batch_size)
    error = error/len(X)

    return loss, error



def evaluate(model, criterion, X, labels, batch_size=None):
    """
    Compute loss and error for the given trained model and data set.

    Args:
          model: Module - Network to train on
          criterion: Module - loss function
          X: tensor - training data
          labels: tensor - train labels
          batch_size: integer - number of samples per batch

    Outputs:
          loss: integer - mean loss over all batches
          error: integer - mean error over all samples
    """
    loss  = 0
    error = 0

    # In case the data is small enough to not require any batch computation we 
    # implement : batch size = data set size
    if batch_size == None:
      batch_size = X.size(0)
    
    for i in range(0, X.size(0), batch_size):
      
        # Select batch
        x = X[i:i+batch_size, :]
        lab = labels[i:i+batch_size]  
      
        # Compute output to obtain loss
        output = model.forward(x)
        loss += criterion.forward(output, lab.long())

        # Compute predictions to obtain errors
        pred = convert_to_predictions(output)
        error += (pred != lab).sum()

    # Average loss over batches and error over samples
    loss = loss/(len(X)/batch_size)
    error = error/len(X)

    return loss, error


   