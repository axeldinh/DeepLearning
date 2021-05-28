import torch # Only used for savings, prints and plottings
from python_scripts import *

# Here we will always load the results for part 1 and 2.

# The trainings are done 1 time on 200 epochs
num_epochs = 200
num_trainings = 1

#######################
# 0-Run a Net with ReLU, MSE and SGD

print("========================================")
print("Part 0: Training a MLP using ReLU as activation, MSE loss and the SGD optimizer:\n")

model_config = [ReLU, 1, 0.0, 'default']
model = create_model(*model_config)
criterion = LossMSE()
optimizer = SGD(model.param(), lr = 0.05)

print("Training...")

train_losses, test_losses, train_errors, test_errors = multiple_training(model_config, criterion, optimizer, num_epochs = num_epochs, num_trainings = num_trainings,
                                                                   display = False)
                                                                   
print("Training over...")

train_errors = torch.Tensor(train_errors)
test_errors = torch.Tensor(test_errors)

# If we train more than once, we print the mean value and the standard deviation of the mean of the best error rates
if num_trainings > 1:
    print("Mean Best Train Error = {:.2f}%(\u00B1{:.2f}), Mean Best Test Error = {:.2f}%(\u00B1{:.2f})".format(train_errors.min(dim = 1).values.mean()*100,
                                                                                                    train_errors.min(dim = 1).values.std()*100,
                                                                                                    test_errors.min(dim = 1).values.mean()*100,
                                                                                                    test_errors.min(dim = 1).values.std()*100))

#######################
# 1-Activations and Initializations
# Compare inits and activations

print("========================================")
print("Part 1: Training a MLP with different activations and weight initializations:")
print("        We use the MSE loss and the SGD optimizer...\n")

print("        For this part, we will load and display results made on 10 rounds to relax the computations:")

activations = [ReLU, Tanh]
init_methods = ['default', 'xavier', 'he']

train_losses = {}
test_losses = {}
train_errors = {}
test_errors = {}

# Load the results

print("Loading the results...")
ActivationsVsInit = torch.load('results/ActivationsVsInit.pt')

# We recover the data from the dict, and print the mean value of the best error rates achieved, along with the standard deviation
best_errors = {}
for activation in activations:
      for init_method in init_methods:
        
            key = [x for x in list(ActivationsVsInit['train_losses'].keys()) if activation.__name__ + init_method in x][0]
            train_losses[key] = torch.Tensor(ActivationsVsInit['train_losses'][key])
            train_errors[key] = torch.Tensor(ActivationsVsInit['train_errors'][key])
            test_losses[key] = torch.Tensor(ActivationsVsInit['test_losses'][key])
            test_errors[key] = torch.Tensor(ActivationsVsInit['test_errors'][key])
            best_errors[key] = test_errors[key].min(dim = 1).values
            
            print("{} with {} initialization:".format(activation.__name__, init_method))
            print("\t\tMean Best Train Error = {:.2f}%(\u00B1{:.2f})".format(train_errors[key].min(dim = 1).values.mean()*100, train_errors[key].min(dim = 1).values.std()*100))
            print("\t\tMean Best Test Error = {:.2f}%(\u00B1{:.2f})".format(test_errors[key].min(dim = 1).values.mean()*100, test_errors[key].min(dim = 1).values.std()*100))

#######################
# 2-Losses and Optimizers
# Compare losses and optimizers, taking ReLU and default parameters' initialization

print("========================================")
print("Part 2: Training a MLP with different losses and optimizers:")
print("        We use ReLU activation and the 'default' initialization...\n")
print("        For this part, we will load and display results made on 10 rounds to relax the computations:")

Losses = ['MSE', 'CrossEntropy']
Optimizers = ['SGD', 'Adam']

# Load the results
print("Loading the results...")
LossesOptim = torch.load('results/LossesOptim.pt')

# We recover the data from the dict, and print the mean value of the best error rates achieved, along with the standard deviation
best_errors = {}
for loss_fn in Losses:
      for optimizer_fn in Optimizers:
        
            key = [x for x in list(LossesOptim['train_losses'].keys()) if loss_fn + optimizer_fn in x][0]
            train_losses[key] = torch.Tensor(LossesOptim['train_losses'][key])
            train_errors[key] = torch.Tensor(LossesOptim['train_errors'][key])
            test_losses[key] = torch.Tensor(LossesOptim['test_losses'][key])
            test_errors[key] = torch.Tensor(LossesOptim['test_errors'][key])
            best_errors[key] = test_errors[key].min(dim = 1).values
            
            print("{} loss with {} optimizer:".format(loss_fn, optimizer_fn))
            print("\t\tMean Best Train Error = {:.2f}%(\u00B1{:.2f})".format(train_errors[key].min(dim = 1).values.mean()*100, train_errors[key].min(dim = 1).values.std()*100))
            print("\t\tMean Best Test Error = {:.2f}%(\u00B1{:.2f})".format(test_errors[key].min(dim = 1).values.mean()*100, test_errors[key].min(dim = 1).values.std()*100))

#######################
# 3-Scheduler
# Look at the loss with a scheduler, with the Cross Entropy loss, the Adam optimizer and 'default' weight initialization

print("========================================")
print("Part 3: Training a MLP with a scheduler, the Cross Entropy loss, the Adam optimizer and 'default' weight initialization:\n")

lr = 0.001

model_config = [ReLU, 2, 0.0, 'default']
model = create_model(*model_config)
criterion = CrossEntropyLoss()
optimizer = Adam(model.param(), lr)
scheduler = ReduceOnPlateau(optimizer, patience = 10, reduce_coef = 0.5, display = False)

print("training...")

_, _, train_error, test_error = multiple_training(model_config, criterion, optimizer, scheduler = scheduler,
                                                                             num_epochs = num_epochs, num_trainings = num_trainings,
                                                                             display = False)

print("Training over...")

train_error = torch.Tensor(train_error)
test_error = torch.Tensor(test_error)

# If we train more than once, we print the mean value and the standard deviation of the mean of the best error rates
if num_trainings > 1:
    
    print("Mean Best Train Error = {:.2f}%(\u00B1{:.2f}), Mean Best Test Error = {:.2f}%(\u00B1{:.2f})".format(train_error.min(dim = 1).values.mean()*100,
                                                                                                    train_error.min(dim = 1).values.std()*100,
                                                                                                    test_error.min(dim = 1).values.mean()*100,
                                                                                                    test_error.min(dim = 1).values.std()*100))
                                                                                                
# Now we compare the losses from this model, and the same one without the scheduler

# Train a model without scheduler

model_config = [ReLU, 2, 0.0, 'default']
model = create_model(*model_config)
criterion = CrossEntropyLoss()
optimizer = Adam(model.param(), lr)

print("\nTraining of the same model without scheduler:")

print("Training...")
_, _, train_error, test_error = multiple_training(model_config, criterion, optimizer, scheduler = None,
                                                  num_epochs = num_epochs, num_trainings = num_trainings,
                                                  display = False)
print("Training over...")

train_error = torch.Tensor(train_error)
test_error = torch.Tensor(test_error)

# If we train more than once, we print the mean value and the standard deviation of the mean of the best error rates
if num_trainings > 1:
    print("Mean Best Train Error = {:.2f}%(\u00B1{:.2f}), Mean Best Test Error = {:.2f}%(\u00B1{:.2f})".format(train_error.min(dim = 1).values.mean()*100,
                                                                                                    train_error.min(dim = 1).values.std()*100,
                                                                                                    test_error.min(dim = 1).values.mean()*100,
                                                                                                    test_error.min(dim = 1).values.std()*100))

#######################
# 4-Dropout
# With the same configuration as part 3 (with scheduler), we test the implementation of Dropout (with p = 0.1)

print("========================================")
print("Part 4: Training a MLP with Dropout (p = 0.1):")
print("        We use a scheduler, the Cross Entropy loss, the Adam optimizer and 'default' weight initialization...\n")

lr = 0.001

model_config = [ReLU, 2, 0.1, 'default']
model = create_model(*model_config)
criterion = CrossEntropyLoss()
optimizer = Adam(model.param(), lr)
scheduler = ReduceOnPlateau(optimizer, patience = 10, reduce_coef = 0.5, display = False)

print("training...")

train_loss, test_loss, train_error, test_error = multiple_training(model_config, criterion, optimizer, scheduler = scheduler,
                                                                             num_epochs = num_epochs, num_trainings = num_trainings,
                                                                             display = False)

print("Training over...")

train_error = torch.Tensor(train_error)
test_error = torch.Tensor(test_error)

# If we train more than once, we print the mean value and the standard deviation of the mean of the best error rates
if num_trainings > 1:
    print("Mean Best Train Error = {:.2f}%(\u00B1{:.2f}), Mean Best Test Error = {:.2f}%(\u00B1{:.2f})".format(train_error.min(dim = 1).values.mean()*100,
                                                                                                    train_error.min(dim = 1).values.std()*100,
                                                                                                    test_error.min(dim = 1).values.mean()*100,
                                                                                                    test_error.min(dim = 1).values.std()*100))