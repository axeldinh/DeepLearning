# This script is the same as test.py, except we also generate the figures and the results (in order to relax the number of coded lines in test.py)

import torch # Only used for savings, prints and plottings
from python_scripts import *

# Warning, if load_1 or load_2 is set to False, the training can be quit long, and it will overwrite previous results
load_1 = True # Allows to load a file containig results from already trained models for part 1
load_2 = True # Allows to load a file containig results from already trained models for part 2

# The trainings are done 1 time on 200 epochs
num_epochs = 200
num_trainings = 1

print("Models are trained {} times on {} epochs".format(num_trainings, num_epochs))

#######################
# 0-Run a Net with ReLU, MSE and SGD
#   Then save the losses in a lineplot

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
                                                                                                
# Plotting of the losses

plt.figure()
plt.title('Losses for ReLU activation, MSE Loss and SGD')
x = range(1, num_epochs+1)
if num_trainings > 1:
    for i in range(num_trainings):
        if i == num_trainings -1:
            plt.plot(x, train_losses[i], alpha = 0.5, color = 'tab:blue', label = 'Train')
            plt.plot(x, test_losses[i], alpha = 0.5, color = 'tab:orange', label = 'Test')
        else:
            plt.plot(x, train_losses[i], alpha = 0.5, color = 'tab:blue')
            plt.plot(x, test_losses[i], alpha = 0.5, color = 'tab:orange')
    plt.plot(x, torch.Tensor(train_losses).mean(dim = 0), color = 'k', label = 'Train Mean')
    plt.plot(x, torch.Tensor(test_losses).mean(dim = 0), color = 'k', ls = '--', label = 'Test Mean')
else:
    plt.plot(x, torch.Tensor(train_losses).mean(dim = 0), label = 'Train')
    plt.plot(x, torch.Tensor(test_losses).mean(dim = 0), label = 'Test')
    
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.savefig('figures/ReLU_MSE_SGD.png', format = 'png')

#######################
# 1-Activations and Initializations
# Compare inits and activations

print("========================================")
print("Part 1: Training a MLP with different activations and weight initializations:")
print("        We use the MSE loss and the SGD optimizer...\n")

activations = [ReLU, Tanh]
init_methods = ['default', 'xavier', 'he']

train_losses = {}
test_losses = {}
train_errors = {}
test_errors = {}

# Put the best learning rates found by grid search
best_lrs = {}
for activation in activations:
    for init_method in init_methods:
        if activation.__name__ == 'ReLU':
            best_lrs[activation.__name__ + init_method] = 0.05
        elif init_method == 'he':
            best_lrs[activation.__name__ + init_method] = 0.01
        else:
            best_lrs[activation.__name__ + init_method] = 0.1

# Make the training or load
if not load_1:
    
    print("Training...")
    
    for activation in activations:
        for init_method in init_methods:
    
            lr = best_lrs[activation.__name__ + init_method]

            print(f"{activation.__name__} and {init_method} initialization, lr = {lr}\nBegin training")

            model_config = [activation, 1, 0.0, init_method]
            model = create_model(*model_config)
            criterion = LossMSE()
            optimizer = SGD(model.param(), lr)
            train_loss, test_loss, train_error, test_error = multiple_training(model_config, criterion, optimizer,
                                                                             num_epochs = num_epochs, num_trainings = num_trainings,
                                                                             display = False)

            train_losses[activation.__name__ + init_method + f"_lr{lr}"] = torch.Tensor(train_loss)
            test_losses[activation.__name__ + init_method + f"_lr{lr}"] = torch.Tensor(test_loss)
            train_errors[activation.__name__ + init_method + f"_lr{lr}"] = torch.Tensor(train_error)
            test_errors[activation.__name__ + init_method + f"_lr{lr}"] = torch.Tensor(test_error)
          
    ActivationsVsInit = {}
    ActivationsVsInit['train_losses'] = train_losses
    ActivationsVsInit['train_errors'] = train_errors
    ActivationsVsInit['test_losses'] = test_losses
    ActivationsVsInit['test_errors'] = test_errors
    torch.save(ActivationsVsInit, 'results/ActivationsVsInit.pt')

print("Loading the results...")
ActivationsVsInit = torch.load('results/ActivationsVsInit.pt')

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

# Make the plot and save it

labels = [x.split('_')[0] for x in list(best_errors.keys())]
labels = [x.replace('ReLU', 'ReLU+') for x in labels]
labels = [x.replace('Tanh', 'Tanh+') for x in labels]

plt.figure(figsize = (10,7))
_ = plt.boxplot([(100*best_errors[key].tolist()) for key in best_errors.keys()], labels=labels)
plt.ylabel('Error Rate [%]')
plt.title('Best Error Rates achieved')

plt.savefig("figures/ActivationsAndInit.png", format = 'png')

#######################
# 2-Losses and Optimizers
# Compare losses and optimizers, taking ReLU and default parameters' initialization

print("========================================")
print("Part 2: Training a MLP with different losses and optimizers:")
print("        We use ReLU activation and the 'default' initialization...\n")

Losses = ['MSE', 'CrossEntropy']
Optimizers = ['SGD', 'Adam']

# Define the learning rates for those parameters
best_lrs = {}
for loss_fn in Losses:
    for optimizer_fn in Optimizers:
        if optimizer_fn == 'SGD':
            best_lrs[loss_fn + optimizer_fn] = 0.05
        elif optimizer_fn == 'Adam':
            best_lrs[loss_fn + optimizer_fn] = 0.001

if not load_2:
    
    print("Training...")
    
    train_losses = {}
    train_errors = {}
    test_losses = {}
    test_errors = {}
    
    for loss_fn in Losses:
        for optimizer_fn in Optimizers:
            
            lr = best_lrs[loss_fn + optimizer_fn]
            
            print(f"{loss_fn} loss and {optimizer_fn} optimizer, lr = {lr}\nBegin training")
            
            if loss_fn == 'MSE':
                model_config = [ReLU, 1, 0.0, 'default']
                model = create_model(*model_config)
                criterion = LossMSE()
            
            elif loss_fn == 'CrossEntropy':
                model_config = [ReLU, 2, 0.0, 'default']
                model = create_model(*model_config)
                criterion = CrossEntropyLoss()
                
            if optimizer_fn == 'SGD':
                optimizer = SGD(model.param(), lr)
            elif optimizer_fn == 'Adam':
                optimizer = Adam(model.param(), lr)
                
            train_loss, test_loss, train_error, test_error = multiple_training(model_config, criterion, optimizer,
                                                                             num_epochs = num_epochs, num_trainings = num_trainings,
                                                                             display = False)
                                                                                 
            train_losses[loss_fn + optimizer_fn] = torch.Tensor(train_loss)
            train_errors[loss_fn + optimizer_fn] = torch.Tensor(train_error)
            test_losses[loss_fn + optimizer_fn] = torch.Tensor(test_loss)
            test_errors[loss_fn + optimizer_fn] = torch.Tensor(test_error)
    
    LossesOptim = {}
    
    LossesOptim['train_losses'] = train_losses
    LossesOptim['train_errors'] = train_errors
    LossesOptim['test_losses'] = test_errors
    LossesOptim['test_errors'] = test_errors
    
    torch.save(LossesOptim, 'results/LossesOptim.pt')

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

# Make the plot and save it

labels = [x.split('_')[0] for x in list(best_errors.keys())]
labels = [x.replace('MSE', 'MSE+') for x in labels]
labels = [x.replace('CrossEntropy', 'CrossEntropy+') for x in labels]

plt.figure(figsize = (10,7))
_ = plt.boxplot([(100*best_errors[key]).tolist()for key in best_errors.keys()], labels=labels)
plt.ylabel('Error Rate [%]')
plt.title('Best Error Rates achieved')

plt.savefig("figures/LossesOptim.png", format = 'png')


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
                                                                                                
# Now we compare the losses from this model, and the same one without the scheduler, we only compare them on one run

sched_train_loss = torch.Tensor(train_loss)[0]
sched_test_loss = torch.Tensor(test_loss)[0]

# Train a model without scheduler

model_config = [ReLU, 2, 0.0, 'default']
model = create_model(*model_config)
criterion = CrossEntropyLoss()
optimizer = Adam(model.param(), lr)

print("\nTraining of the same model without scheduler:")

print("Training...")
no_sched_train_loss, no_sched_test_loss, train_error, test_error = multiple_training(model_config, criterion, optimizer, scheduler = None,
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

no_sched_train_loss =  torch.Tensor(no_sched_train_loss[0])
no_sched_test_loss =  torch.Tensor(no_sched_test_loss[0])

# Plotting of the losses

x = range(1, num_epochs+1)
plt.figure()
plt.title('Train Losses with or without scheduler')
plt.plot(x, sched_train_loss, label = 'With Scheduler')
plt.plot(x, no_sched_train_loss, label = 'Without Scheduler')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.legend()
plt.ylim([0,0.1])

plt.savefig('figures/SchedulerTrainLoss.png', format = 'png')

x = range(1, num_epochs+1)
plt.figure()
plt.title('Test Losses with or without scheduler')
plt.plot(x, sched_test_loss, label = 'With Scheduler')
plt.plot(x, no_sched_test_loss, label = 'Without Scheduler')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.legend()
plt.ylim([0,0.1])

plt.savefig('figures/SchedulerTestLoss.png', format = 'png')

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

# Plotting of the losses

train_loss = torch.Tensor(train_loss)
test_loss = torch.Tensor(test_loss)

plt.figure()
plt.title('Losses with Dropout')
x = range(1, num_epochs+1)
plt.plot(x, train_loss[0], label = 'Train')
plt.plot(x, test_loss[0], label = 'Test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.savefig('figures/dropout.png', format = 'png')