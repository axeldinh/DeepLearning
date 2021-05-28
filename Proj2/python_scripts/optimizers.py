from torch import empty
import math

class Optimizer(object):
    """
    Super-class for optimizers and schedulers.
    """
    
    def step(self, *inputs):
        """
        Updates the gradients, or the learning rate if it is used for a scheduler
        """
        raise NotImplementedError


#### OPTIMIZERS ######################

### Contains 2 optimizers: SGD and Adam

class SGD(Optimizer):
    """
    Applies stochastic gradient descent.

    Attributes:
          - lr (float): learning rate
          - parameters (list): contains all parameters from the associated model.
          - momentum (float): momentum factor
          - l2_penalty (float): l2 penalty factor (substracts l2_penalty*lr to the weights at the update)
    """
    def __init__(self, parameters, lr = 0.01, momentum = 0.0, l2_penalty = 0.0):
        
        super(SGD, self).__init__()
        
        self.lr = lr
        self.parameters = parameters
        self.momentum = momentum
        self.l2_penalty = l2_penalty
        
        # Initialize the velocities for momentum
        self.velocities = [empty(w.size()).fill_(0.) for (w, _) in self.parameters]
        
    def step(self):
        
        for (weight, grad), velocity in zip(self.parameters, self.velocities):
            # Compute the penalty term
            penalty = self.l2_penalty * weight
            # Compute the velocities
            velocity.mul_(self.momentum).add_(self.lr * grad)
            # Update the weights
            weight.sub_(velocity + self.lr*penalty)
            
class Adam(Optimizer):
    """
    Applies Adam.

    Attributes:
          - lr (float): learning rate
          - parameters (list): contains all parameters from the associated model.
          - momentum (float): momentum factor
          - l2_penalty (float): l2 penalty factor (substracts l2_penalty*lr to the weights at the update)
    """
    def __init__(self, parameters, lr = 0.01, betas = (0.9, 0.999), eps = 1e-08, l2_penalty = 0.0):
        
        super(Adam, self).__init__()
        
        self.lr = lr
        self.parameters = parameters
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.l2_penalty = l2_penalty
        self.steps = 0 #Needed to keep track of the progress
        
        # Create the intermediate steps of Adam
        self.momentums = [empty(w.size()).fill_(0.) for (w, _) in self.parameters]
        self.velocities = [empty(w.size()).fill_(0.) for (w, _) in self.parameters]
        
    def step(self):
        
        self.steps += 1
        
        for i, (weight, grad) in enumerate(self.parameters):
            # Compute the penalty term
            penalty = self.l2_penalty * weight
            # Compute the momentums and velocities
            self.momentums[i] = self.beta1 * self.momentums[i] + (1 - self.beta1)*grad
            self.velocities[i] = self.beta2 * self.velocities[i] + (1-self.beta2)*grad.pow(2)
            
            m_hat = self.momentums[i] / (1-math.pow(self.beta1, self.steps))
            v_hat = self.velocities[i] / (1-math.pow(self.beta2, self.steps))
            
            # Update the weights
            weight.sub_(self.lr* (m_hat.div(v_hat.sqrt() + self.eps) + penalty))
            
            
#### SCHEDULERS ########################

### Contains 1 scheduler: ReduceOnPlateau

class ReduceOnPlateau(Optimizer):
    """
    Scheduler, will reduce the learning rate by reduce_factor if the loss has not decreased after patience number of  epochs.
    Parameters:
        -optimizer (Optimizer): optimizer where the learning rate update will happen
        -patience (int): number of epoch we will wait before updating the learning rate
        -reduce_coef (float): factor that will multiply the learning rate
        -min_lr (float): if learning rate < min_lr, the scheduler won't do the update
        -threshold (float): what the difference between the last update and the last best one must be to consider it as an improvement
        -display (bool): Wheter or not to print when the scheduler makes an update.
    """
    
    def __init__(
      self, optimizer, patience = 10, reduce_coef = 0.1, min_lr = 0,
      threshold = 1e-4, display = True):
        
        super(ReduceOnPlateau, self).__init__()
        
        self.optimizer = optimizer
        self.patience = patience
        self.reduce_coef = reduce_coef
        self.min_lr = min_lr
        self.threshold = threshold
        self.display = display

        self.last_decrease = 0
        self.last_best_loss = 0
        self.current_epoch = 0
        
    def step(self, loss):
        
        self.current_epoch += 1

        # If it is the first call we just update self.last_best_loss
        if self.current_epoch == 1:
            self.last_best_loss = loss
            return
        
        # If the lr is too small, do nothing
        if self.optimizer.lr <= self.min_lr:
          return
        
        # If we did not decrease the loss, add 1 to the count
        if (loss >= self.last_best_loss - self.threshold):
            self.last_decrease += 1
        
        # If we did, replace the best loss and restart the count
        else:
            self.last_best_loss = loss
            self.last_decrease = 0
        
        # If we've waited enough, multiply the lr by reduce_coef
        if self.patience == self.last_decrease:
            self.optimizer.lr = self.optimizer.lr * self.reduce_coef
            self.last_decrease = 0
            if self.display:
                print(f"Learning rate reduced by a factor {self.reduce_coef} at epoch {self.current_epoch}")
                print(f"Current learning rate = {self.optimizer.lr}")
                
    def init(self):
        """
        Reinitialize the scheduler's parameters for a new training. It leaves most of it unchanged.
        """
        
        self.last_decrease = 0
        self.last_best_loss = 0
        self.current_epoch = 0